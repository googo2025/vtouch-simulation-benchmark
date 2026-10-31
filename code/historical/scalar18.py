"""Unknown-ROI whole-image energy and edge-displacement descriptors.

Local linear optical-flow projection is a matched subspace statistic, not
measured displacement or ground-truth localization. All pixels are searched.
"""
import math
import numpy as np
from v2_full_scene_direct_radiance_v1 import gaussian

SIZES=(4,8,16)

def boxsum(a,size):
    half=size//2;pad=np.pad(a,((half,half),(half,half)))
    integral=np.pad(pad.cumsum(0).cumsum(1),((1,0),(1,0)))
    out=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
    return out[:a.shape[0],:a.shape[1]]

def setup(reference,full_well):
    variance=(255/full_well)**2*(np.maximum(reference,0)*full_well/255+9)*(1+1/16)+(1+1/16)/12
    smooth=gaussian(reference,.6);gy,gx=np.gradient(smooth);sd=np.sqrt(variance)
    gx=gx/sd;gy=gy/sd;grams=[]
    for size in SIZES:
        A=boxsum(gx*gx,size);B=boxsum(gx*gy,size);C=boxsum(gy*gy,size)
        ridge=1e-10*(A+C)+1e-12;A+=ridge;C+=ridge
        grams.append((A,B,C,A*C-B*B))
    return sd,gx,gy,grams

def transform(image,reference,prepared):
    sd,gx,gy,grams=prepared;delta=(image.astype(float)-reference)/sd
    xx=gx*delta;yy=gy*delta;energy=delta*delta-1;features=[]
    for size,(A,B,C,det) in zip(SIZES,grams):
        bx=boxsum(xx,size);by=boxsum(yy,size)
        score=(C*bx*bx-2*B*bx*by+A*by*by)/det
        e=boxsum(energy,size)/math.sqrt(2*size*size)
        index=int(np.argmax(score));y,x=np.unravel_index(index,score.shape)
        # Top32 is a fixed full-image robust-peak descriptor, not selected by truth.
        top32=float(np.partition(score.ravel(),-32)[-32:].mean())
        dx=(C[y,x]*bx[y,x]-B[y,x]*by[y,x])/det[y,x]
        dy=(A[y,x]*by[y,x]-B[y,x]*bx[y,x])/det[y,x]
        values=np.array([e.max(),score[y,x],top32,e[y,x],dx,dy])
        features.extend((np.sign(values)*np.log1p(abs(values))).tolist())
    out=np.array(features);assert np.all(np.isfinite(out));return out

def unit_audit():
    y,x=np.meshgrid(np.arange(48),np.arange(64),indexing='ij')
    reference=80+20*np.sin(x/5)+15*np.cos(y/7);prepared=setup(reference,20000)
    zero=transform(reference,reference,prepared);assert np.max(abs(zero[[1,2,4,5,7,8,10,11,13,14,16,17]]))==0
    image=reference+.08*np.gradient(reference)[1]-.04*np.gradient(reference)[0]
    actual=transform(image,reference,prepared);assert np.max(actual[[1,7,13]])>0
    # Independent dense two-variable least-squares for a declared interior box.
    sd,gx,gy,grams=prepared;size=8;cy,cx=24,32;sl=(slice(cy-4,cy+4),slice(cx-4,cx+4))
    d=(image-reference)/sd;M=np.c_[gx[sl].ravel(),gy[sl].ravel()];b=M.T@d[sl].ravel();H=M.T@M
    ridge=1e-10*np.trace(H)+1e-12;expected=b@np.linalg.solve(H+ridge*np.eye(2),b)
    A,B,C,det=grams[1];bx=boxsum(gx*d,size)[cy,cx];by=boxsum(gy*d,size)[cy,cx]
    got=(C[cy,cx]*bx*bx-2*B[cy,cx]*bx*by+A[cy,cx]*by*by)/det[cy,cx]
    assert abs(got-expected)<1e-10
    return {'zero_displacement_projection_exact_zero':True,'independent_dense_projection_absolute_error':float(abs(got-expected)),
            'limits':'Numerical feature unit test only, not a sensor experiment or performance score.'}
