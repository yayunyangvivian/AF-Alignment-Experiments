from common import *
import torch, argparse, time
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
torch.set_num_threads(2)
P=ECH=FEATURES=OLD=None
REGISTRY=[]
from types import SimpleNamespace
from af_alignment import v2b,scoring
C=SimpleNamespace(E=SimpleNamespace(T=v2b,metrics=scoring.metrics))
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from evaluate_frozen import load_model as load_frozen
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
def load_model(seed,arm,dose):
 expected=0 if arm=='none' else dose
 found=[r for r in REGISTRY if int(r['seed'])==seed and r['arm']==arm and float(r['dose'])==expected]
 assert len(found)==1,(seed,arm,dose)
 return load_frozen(found[0]['directory'],seed,arm)
def main(dose):
 start=time.time();feature=np.load(FEATURES);y=feature['label'];xx=feature['ecg'];meta=feature['metadata'];conditions=feature['conditions'];assert len(y)==10338
 xp=np.load(P/'emb_ecg.npy');mp=np.load(P/'tab.npy');xe=np.load(ECH/'emb_echo.npy');tr=np.loadtxt(P/'ids_train.txt',dtype=int);va=np.loadtxt(P/'ids_val.txt',dtype=int);et=np.loadtxt(ECH/'ids_echo_train.txt',dtype=int);ev=np.loadtxt(ECH/'ids_echo_val.txt',dtype=int)
 rg=np.random.default_rng(20260910);pv=rg.choice(va,512,False);evi=rg.choice(ev,512,False);pt=rg.choice(tr,512,False);eti=rg.choice(et,512,False)
 np.savez(R/'alignment_subset_ids.npz',ptb_val=pv,echo_val=evi,ptb_train=pt,echo_train=eti)
 tag='base' if dose<.3 else 'dose03';out=R/'representations'/tag;out.mkdir(parents=True,exist_ok=True)
 statsrows=[];moves=[];align=[];metrics=[];checks=[];rawarr=[];calarr=[];se=[];ar=[];co=[];spectra=[];probe_spectra=[];ident=[]
 old=np.load(OLD)
 for seed in SEEDS:
  ref,_,_=load_model(seed,'none',0);ht=forward(ref,xp[pt],mp[pt]);he=forward(ref,xe[eti],echo=True);sigma=float(C.E.T._median_sigma(torch.from_numpy(ht),torch.from_numpy(he)));assert sigma>0
  for arm in (ARMS if dose<.3 else ['symmetric','anchored']):
   lam=0. if arm=='none' else dose
   net,pp,ck=load_model(seed,arm,dose);hs=[]
   for j,condition in enumerate(conditions):
    h=forward(net,xx[:,j],meta);hs.append(h);q,c=predict(h,pp)
    if dose<.3:
     ix=np.where((old['seed']==seed)&(old['arm']==arm)&(old['condition']==condition))[0].item();error=max(np.max(np.abs(q-old['raw_probability'][ix])),np.max(np.abs(c-old['calibrated_probability'][ix])));assert error<1e-10,(seed,arm,condition,error)
    else:error=np.nan
    checks.append(dict(seed=seed,arm=arm,dose=lam,condition=condition,prediction_max_abs_difference=error))
    z=transformed(h,pp);s,l=spectrum(h);ps,pl=spectrum(z);sp=separation(h,y);zp=separation(z,y)
    statsrows.append(dict(seed=seed,arm=arm,dose=lam,condition=condition,**s,**sp,**{'probe_'+k:v for k,v in {**ps,**zp}.items()}));spectra.append(l);probe_spectra.append(pl);ident.append(f'{seed}_{arm}_{lam}_{condition}')
    if j==0:h0=h;z0=z
    else:moves.append(dict(seed=seed,arm=arm,dose=lam,condition=condition,**movement(h0,h),**{'probe_'+k:v for k,v in movement(z0,z).items()}))
    metrics.append(dict(seed=seed,arm=arm,dose=lam,condition=condition,**C.E.metrics(y,q,c)));rawarr.append(q);calarr.append(c);se.append(seed);ar.append(arm);co.append(str(condition))
   np.savez_compressed(out/f'seed{seed}_{arm}.npz',h=np.stack(hs),conditions=conditions,record_id=feature['record_id'],label=y)
   hv=forward(net,xp[pv],mp[pv]);eh=forward(net,xe[evi],echo=True);adaptive=float(C.E.T._median_sigma(torch.from_numpy(hv),torch.from_numpy(eh)));lm=float(C.E.T.mean_match_loss(torch.from_numpy(hv),torch.from_numpy(eh)));lc=float(C.E.T.coral_loss(torch.from_numpy(hv),torch.from_numpy(eh)));a,_=spectrum(hv);e,_=spectrum(eh)
   align.append(dict(seed=seed,arm=arm,dose=lam,mean_mismatch=lm,CORAL=lc,MMD_adaptive=kernel_mmd(hv,eh,adaptive),MMD_common=kernel_mmd(hv,eh,sigma),adaptive_bandwidth=adaptive,common_bandwidth=sigma,**{'ptb_'+k:v for k,v in a.items()},**{'echo_'+k:v for k,v in e.items()}))
   print(tag,seed,arm,'complete',round(time.time()-start,1),'seconds',flush=True)
 table(f'representation_stats_{tag}.csv',statsrows);table(f'representation_movement_{tag}.csv',moves);table(f'alignment_diagnostics_{tag}.csv',align);table(f'forward_metrics_{tag}.csv',metrics);table(f'forward_verification_{tag}.csv',checks)
 np.savez_compressed(R/f'spectra_{tag}.npz',eigenvalues=spectra,model_condition=ident)
 np.savez_compressed(R/f'probe_spectra_{tag}.npz',eigenvalues=probe_spectra,model_condition=ident)
 np.savez_compressed(R/f'predictions_{tag}.npz',raw_probability=rawarr,calibrated_probability=calarr,seed=se,arm=ar,dose=[0 if arm=='none' else dose for arm in ar],condition=co,label=y,record_id=feature['record_id'])
 save(f'REPRESENTATIONS_{tag.upper()}_COMPLETE.json',dict(models=30 if dose<.3 else 20,conditions=9,seconds=time.time()-start,records=len(y),classifier_fits=0))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dose',type=float,choices=[.1,.3],default=.1);ap.add_argument('--models',required=True);ap.add_argument('--ptb-dir',required=True);ap.add_argument('--echo-dir',required=True);ap.add_argument('--features',required=True);ap.add_argument('--reference-predictions',required=True);a=ap.parse_args()
 P=Path(a.ptb_dir);ECH=Path(a.echo_dir);FEATURES=Path(a.features);OLD=Path(a.reference_predictions)
 REGISTRY=json.loads(Path(a.models).read_text())
 for r in REGISTRY:r['directory']=str(Path(a.models).resolve().parent/r['directory'])
 main(a.dose)
