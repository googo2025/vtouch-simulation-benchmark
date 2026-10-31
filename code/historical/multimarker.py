"""All manufactured REST landmarks, actual image projections, unknown touch.

Local affine-gradient/nuisance subspaces at EVERY visible REST marker, not
the touched ROI. Coefficients describe images; not true FE marker coordinates,
physical displacement, strain or calibrated marker tracking precision.
"""
import numpy as np
from v2_full_scene_direct_radiance_v1 import gaussian

SIZES=(9,17,25)

def setup(reference,pixels,full_well=20000):
    assert reference.ndim==2 and pixels.ndim==2 and pixels.shape[1]==2
    variance=(255/full_well)**2*(np.maximum(reference,0)*full_well/255+9)*(1+1/16)+(1+1/16)/12
    smooth=gaussian(reference,.6);gy,gx=np.gradient(smooth);h,w=reference.shape;groups=[]
    for size in SIZES:
        half=size//2;yy,xx=np.meshgrid(np.arange(-half,half+1),np.arange(-half,half+1),indexing='ij')
        x=np.rint(pixels[:,0]-.5).astype(int)[:,None]+xx.ravel()[None]
        y=np.rint(pixels[:,1]-.5).astype(int)[:,None]+yy.ravel()[None]
        valid=(x>=0)&(x<w)&(y>=0)&(y<h);x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
        sd=np.sqrt(variance[y,x]);a=gx[y,x]/sd;b=gy[y,x]/sd
        u=xx.ravel()[None]/half;v=yy.ravel()[None]/half
        B=np.stack((a,b,u*a,v*b,-v*a+u*b,1/sd,smooth[y,x]/sd),axis=2)*valid[:,:,None]
        norm=np.maximum(np.sqrt((B*B).sum(1)),1e-8);B=B/norm[:,None]
        gram=np.einsum('npi,npj->nij',B,B)
        cholesky=np.linalg.cholesky(gram+1e-6*np.eye(7)[None])
        groups.append(dict(x=x,y=y,valid=valid,sd=sd,B=B,cholesky=cholesky,reference=reference[y,x],
            pixel_count=valid.sum(1),normalized_basis_gram=gram,column_norm=norm))
    return dict(groups=groups,marker_pixels=pixels.copy(),marker_count=len(pixels),feature_count=len(pixels)*9*len(SIZES))

def transform(image,prepared):
    features=[]
    for group in prepared['groups']:
        delta=(image[group['y'],group['x']].astype(float)-group['reference'])/group['sd']*group['valid']
        rhs=np.einsum('npi,np->ni',group['B'],delta)
        projected=np.linalg.solve(group['cholesky'],rhs[:,:,None])[:,:,0]
        total=(delta*delta).sum(1);fitted=(projected*projected).sum(1);count=group['pixel_count']
        # Empirical null calibration below, not an exact chi-square probability claim.
        energy=(total-count)/np.sqrt(np.maximum(2*count,1))
        remainder=(total-fitted-np.maximum(count-7,0))/np.sqrt(np.maximum(2*(count-7),1))
        values=np.c_[projected,energy,remainder]
        features.append((np.sign(values)*np.log1p(abs(values))).ravel())
    result=np.concatenate(features);assert result.shape==(prepared['feature_count'],) and np.all(np.isfinite(result));return result

def unit_audit():
    y,x=np.meshgrid(np.arange(64),np.arange(80),indexing='ij');reference=70+20*np.sin(x/5)+15*np.cos(y/7)
    pixels=np.array([[30.5,30.5],[50.5,32.5]]);prepared=setup(reference,pixels)
    image=reference+.08*np.gradient(reference)[1]-.04*np.gradient(reference)[0]
    out=transform(image,prepared);assert len(out)==54
    group=prepared['groups'][1];delta=(image[group['y'],group['x']]-group['reference'])/group['sd']*group['valid']
    B=group['B'][0];H=B.T@B+1e-6*np.eye(7);b=B.T@delta[0]
    expected=float(b@np.linalg.solve(H,b));projection=np.linalg.solve(group['cholesky'][0],b)
    error=abs(expected-float(projection@projection));assert error<1e-8
    zero=transform(reference,prepared)
    for groupindex in range(3):assert np.all(zero[groupindex*18:groupindex*18+18].reshape(2,9)[:,:7]==0)
    return dict(status='pass_local_affine_projection_unit',direct_regularized_projection_energy_error=error,
        exact_reference_has_zero_affine_coefficients=True,marker_count=2,dimensions=54,
        limits='Image-statistic numerical identity only, not a true mechanical displacement estimator or sensor performance result.')
