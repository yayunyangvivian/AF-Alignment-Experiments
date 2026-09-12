import numpy as np,wfdb,warnings
from scipy.signal import resample_poly
from . import georgia_features as F
CONDITIONS=['clean','noise_005','noise_015','wander_010','wander_030','amplitude_080','amplitude_120','mask_V1','mask_V1_V2']
def one(row):
 p=F.DATA/row.relative_path;h=wfdb.rdheader(str(p.with_suffix('')));sig,_=wfdb.rdsamp(str(p.with_suffix('')))
 assert h.fs==500 and np.isfinite(sig).all()
 order=[s.upper() for s in h.sig_name];sig=sig[:,[order.index(s) for s in F.LEADS]]
 rms=np.sqrt(np.mean((sig-sig.mean(axis=0))**2,axis=0));rng=np.random.default_rng(np.random.SeedSequence([20260910,int(row.record_id[1:])]))
 noise=rng.standard_normal(sig.shape)*rms;phase=rng.uniform(0,2*np.pi,12)
 wander=np.sin(2*np.pi*.3*np.arange(len(sig))[:,None]/500+phase)*rms
 v1=sig.copy();v1[:,6]=0;v12=sig.copy();v12[:,6:8]=0
 variants=[sig,sig+.05*noise,sig+.15*noise,sig+.1*wander,sig+.3*wander,.8*sig,1.2*sig,v1,v12]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  out=np.stack([F.T.ecg_features_12lead(resample_poly(v,1,5,axis=0),100) for v in variants])
 assert np.isfinite(out).all(),row.record_id
 return out
