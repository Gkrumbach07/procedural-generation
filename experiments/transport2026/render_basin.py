import numpy as np
from PIL import Image
z0=np.load('basin_z0.npy'); z1=np.load('basin_out.npy'); m=np.load('basin_m.npy'); hf=np.load('basin_h_f.npy')
dx=12.5
def hillshade(a,az=315.,alt=40.):
    gy,gx=np.gradient(a,dx); slope=np.arctan(np.hypot(gx,gy)); asp=np.arctan2(-gx,gy)
    az_,alt_=np.radians(az),np.radians(alt)
    return np.clip(np.sin(alt_)*np.cos(slope)+np.cos(alt_)*np.sin(slope)*np.cos(az_-asp),0,1)
lo,hi=np.percentile(z0[m==1],1),np.percentile(z0[m==1],99)
def img(a,water=None):
    t=np.clip((a-lo)/max(hi-lo,1e-9),0,1); hs=hillshade(a)
    rgb=np.stack([0.22+0.68*t,0.32+0.56*t,0.28+0.44*t],-1)*(0.30+0.70*hs)[...,None]
    if water is not None:
        w=np.log1p(np.maximum(water,0)/max(water.max(),1e-30)*1e4)
        wm=(w>np.percentile(w[m==1],96))&(m==1)
        rgb[wm]=rgb[wm]*0.2+np.array([0.12,0.32,0.85])*0.8
    rgb[m==0]*=0.35
    return (np.clip(rgb,0,1)*255).astype(np.uint8)
a=img(z0); b=img(z1,hf)
gap=np.full((a.shape[0],8,3),255,np.uint8)
out=np.concatenate([a,gap,b],1)
im=Image.fromarray(out); im=im.resize((im.width*2,im.height*2),Image.LANCZOS)
im.save('basin_cmp.png')
print('left = baked input, right = after 2026 transport method')
print('relief %.1f -> %.1f m | mean |grad| %.4f -> %.4f'%(np.ptp(z0[m==1]),np.ptp(z1[m==1]),
      np.hypot(*np.gradient(z0,dx)[::-1])[m==1].mean(),np.hypot(*np.gradient(z1,dx)[::-1])[m==1].mean()))
