"""Robust low-frequency illumination alignment, fitted to a normal-only template."""
import numpy as np

def align_illumination(image, reference, max_correction=.25):
    if image.shape != reference.shape:
        raise ValueError("image/reference shapes differ")
    h,w,_=image.shape
    y,x=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    design=np.stack([np.ones_like(x),x,y],-1).reshape(-1,3)
    residual=(image-reference).reshape(-1,3)
    weights=np.ones(h*w)
    for _ in range(4):
        root=np.sqrt(weights)[:,None]
        coeff=np.linalg.lstsq(design*root,residual*root,rcond=None)[0]
        error=np.sqrt(np.mean((residual-design@coeff)**2,axis=1))
        scale=max(float(np.median(error))*1.4826,.01)
        weights=np.minimum(1,1.5*scale/np.maximum(error,1e-8))
    field=np.clip((design@coeff).reshape(h,w,3),-max_correction,max_correction)
    return np.clip(image-field,0,1).astype(np.float32)
