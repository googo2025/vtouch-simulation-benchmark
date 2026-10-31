"""Actual NumPy nearest-template, ridge and MLP training/inference.

No external ML package is required. PCA/normalization are training-only.
These small development models do not prove full-surface generalization.
"""
import numpy as np


def fit_pca(x,components=16):
    mean=x.mean(0);centered=x-mean
    # Thin train Gram eigenproblem avoids repeated wide-image SVD. No validation
    # pixels, labels or statistics enter the fitted projection.
    values,u=np.linalg.eigh(centered@centered.T);order=np.argsort(values)[::-1]
    order=order[values[order]>max(1e-12,float(values.max())*1e-12)][:components]
    if not len(order):
        return dict(mean=mean,basis=np.zeros((1,x.shape[1])),scale=np.ones(1))
    s=np.sqrt(values[order]);v=(u[:,order].T@centered)/s[:,None]
    scale=np.maximum(s/np.sqrt(max(1,len(x)-1)),1e-3)
    return dict(mean=mean,basis=v,scale=scale)


def encode(x,pca):return ((x-pca['mean'])@pca['basis'].T)/pca['scale']


def train(x,y,kind,seed=29,classification=False):
    if kind=='nearest':return dict(kind=kind,x=x.copy(),y=y.copy(),classification=classification)
    xx=np.c_[x,np.ones(len(x))]
    if kind=='ridge':
        target=np.eye(int(y.max())+1)[y.astype(int)] if classification else y
        penalty=np.eye(xx.shape[1]);penalty[-1,-1]=0
        return dict(kind=kind,weights=np.linalg.solve(xx.T@xx+.5*penalty,xx.T@target),classification=classification)
    if kind!='mlp':raise ValueError(kind)
    rng=np.random.default_rng(seed);outputs=int(y.max())+1 if classification else y.shape[1]
    hidden=24;W=rng.normal(0,1/np.sqrt(max(1,x.shape[1])),size=(x.shape[1],hidden));b=np.zeros(hidden)
    V=rng.normal(0,.15,size=(hidden,outputs));c=np.zeros(outputs);params=[W,b,V,c];m=[np.zeros_like(p) for p in params];v=[p.copy() for p in m]
    target=np.eye(outputs)[y.astype(int)] if classification else y;loss=[]
    for step in range(1,301):
        z=np.tanh(x@W+b);out=z@V+c
        if classification:
            prob=np.exp(out-out.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            error=(prob-target)/len(x);value=float(-np.mean(np.log(np.maximum(prob[np.arange(len(x)),y.astype(int)],1e-12))))
        else:error=2*(out-target)/len(x);value=float(np.mean((out-target)**2))
        gz=(error@V.T)*(1-z*z);grads=[x.T@gz+1e-3*W,gz.sum(0),z.T@error+1e-3*V,error.sum(0)]
        for i,(p,g) in enumerate(zip(params,grads)):
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
            p-=.008*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        if step%20==0:loss.append(value)
    return dict(kind=kind,W=W,b=b,V=V,c=c,classification=classification,training_loss=np.array(loss))


def predict(x,model):
    if model['kind']=='nearest':
        distances=((x[:,None,:]-model['x'][None,:,:])**2).sum(2)
        return model['y'][np.argmin(distances,axis=1)]
    out=np.c_[x,np.ones(len(x))]@model['weights'] if model['kind']=='ridge' else np.tanh(x@model['W']+model['b'])@model['V']+model['c']
    return np.argmax(out,axis=1) if model['classification'] else out


def serial(model):return {k:v for k,v in model.items() if isinstance(v,np.ndarray)}
