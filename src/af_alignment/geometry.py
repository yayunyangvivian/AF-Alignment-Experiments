import numpy as np
import torch
from scipy.special import expit
def forward(net,x,meta=None,echo=False):
 with torch.no_grad():
  return np.concatenate([net(**({'z_echo':torch.from_numpy(x[i:i+1024])} if echo else {'z_ecg':torch.from_numpy(x[i:i+1024]),'z_meta':torch.from_numpy(meta[i:i+1024])}))[0].numpy() for i in range(0,len(x),1024)])

def transformed(h,p):
 z=h/(np.linalg.norm(h,axis=1,keepdims=True)+1e-8);z-=p['scaler_mean'].astype(z.dtype);z/=p['scaler_scale'].astype(z.dtype);return z

def predict(h,p):
 z=transformed(h,p);raw=expit((z@p['probe_coef'].T+p['probe_intercept']).ravel());cal=expit((raw[:,None]@p['calibrator_coef'].T+p['calibrator_intercept']).ravel());return raw,cal

def separation(h,y):
 h=np.asarray(h,float);a=h[y==0];b=h[y==1];cent=float(np.sum((a.mean(0)-b.mean(0))**2));w0=float(np.mean(np.sum((a-a.mean(0))**2,axis=1)));w1=float(np.mean(np.sum((b-b.mean(0))**2,axis=1)));w=.5*(w0+w1)
 return dict(centroid_distance=np.sqrt(cent),centroid_distance_sq=cent,within_class0=w0,within_class1=w1,balanced_within=w,separation_ratio=cent/w if w else np.nan)

def spectrum(h):
 z=np.asarray(h,float);z-=z.mean(0);cov=z.T@z/(len(z)-1);lam=np.linalg.eigvalsh(cov);assert lam.min()>-1e-7;lam=lam.clip(0);trace=lam.sum()
 return dict(trace=trace,PR=trace**2/(lam@lam) if trace else 0,PC1_share=lam[-1]/trace if trace else np.nan,rms_pairwise_distance=np.sqrt(2*trace)),lam

def movement(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);dist=np.sum((a-b)**2,1);radius=np.mean(np.sum((a-a.mean(0))**2,1));cos=np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
 return dict(mean_distance=np.mean(np.sqrt(dist)),rms_relative=np.sqrt(dist.mean()/radius) if radius else np.nan,mean_cosine_distance=np.mean(1-np.clip(cos,-1,1)))

def kernel_mmd(a,b,sigma):
 a=torch.from_numpy(a);b=torch.from_numpy(b);aa=torch.cdist(a,a)**2;bb=torch.cdist(b,b)**2;ab=torch.cdist(a,b)**2;ans=0.
 for mult in [.5,1.,2.,4.]:
  g=1/(2*(sigma*mult)**2+1e-12);ka=torch.exp(-g*aa);kb=torch.exp(-g*bb);kc=torch.exp(-g*ab);ans+=((ka.sum()-ka.diag().sum())/(len(a)*(len(a)-1))+(kb.sum()-kb.diag().sum())/(len(b)*(len(b)-1))-2*kc.mean()).item()
 return ans
