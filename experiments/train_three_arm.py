from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
#!/usr/bin/env python3
"""New matched experiment using retained model/loss definitions; not historical recovery.
Regularization is PTB-only in ALL arms to permit a literal PTB-only control.
Test evaluation happens only after all validation-selected fits are saved.
"""
import os,sys,argparse,json,hashlib,time,importlib.util,copy
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import platform, scipy, sklearn
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):Path(p).write_text(json.dumps(z,indent=2,default=str)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ptb-dir',required=True);ap.add_argument('--echo-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--hours',type=float,default=2);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--run',action='store_true');a=ap.parse_args()
 from af_alignment import v2b as T
 from af_alignment.runtime import ROOT,check_inputs,output_dir
 ptb=Path(a.ptb_dir);echo=Path(a.echo_dir);check_inputs(ptb,echo)
 protocol=json.loads((ROOT/'configs/three_arm.json').read_text());a.seeds=protocol['seeds'];a.epochs=protocol['epochs'];source=Path(T.__file__)
 if not a.run:print('Retained input hashes verified; no training.');return
 out=output_dir(a.output);torch.set_num_threads(a.threads)
 splits={s:np.loadtxt(ptb/f'ids_{s}.txt',dtype=int) for s in ['train','val','test']}
 x=np.load(ptb/'emb_ecg.npy').astype('float32');meta=np.load(ptb/'tab.npy').astype('float32');y=np.load(ptb/'y_af.npy').astype('float32');patients=np.load(ptb/'test_patient_ids.npy')
 ex=np.load(echo/'emb_echo.npy').astype('float32');ey=np.load(echo/'y_ef.npy').astype('float32');ei=np.loadtxt(echo/'ids_echo_train.txt',dtype=int)
 assert len(patients)==len(splits['test']);assert not any(set(splits[u])&set(splits[v]) for u,v in [('train','val'),('train','test'),('val','test')])
 cfg=dict(seeds=a.seeds,epochs=a.epochs,dimension=128,proj_hidden=256,nhead=4,layers=2,ff_dim=256,head_hidden=128,dropout=.1,ptb_batch=1024,echo_batch=256,ptb_steps_per_epoch=len(splits['train'])//1024,lr=.001,weight_decay=.0001,grad_clip=1,ptb_positive_weight=None,mask_probability=.15,metadata_dropout=.5,slot_permutation=True,EF_standardization='Echo training mean and SD',variance_weight=.5,covariance_weight=1,variance_gamma=1,regularization_cohort='PTB only in all three arms',alignment_weights=dict(mean=.1,coral=.1,mmd=.1),warmup_epochs=5,selection='maximum validation AF head AUC; earliest epoch on ties',probe='row normalization then training StandardScaler then balanced LogisticRegression max_iter=5000',arms=['ptb_only','ptb_ef','ptb_ef_alignment'],pairing='same seed and initialization; same PTB minibatch schedules; subsequent stochastic draws may diverge; no precision benefit assumed',source=str(source),source_sha256=sha(source),runner_sha256=sha(__file__),torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform(),threads=a.threads,max_hours=a.hours)
 inputpaths=[ptb/n for n in ['emb_ecg.npy','tab.npy','y_af.npy','ids_train.txt','ids_val.txt','ids_test.txt','test_patient_ids.npy']]+[echo/n for n in ['emb_echo.npy','y_ef.npy','ids_echo_train.txt']]
 cfg['inputs']={str(p):sha(p) for p in inputpaths}
 cfg['variance_penalty']='mean squared positive part of 1 - sqrt(biased batch variance + 1e-4)'
 cfg['covariance_penalty']='sum squared off-diagonal sample covariances divided by latent dimension'
 cfg['ptb_only_cohort_use']='No Echo observations enter the PTB-only forward pass or objective; common model dimensionality is fixed across arms.'
 if not a.run:
  save(out/'proposed_configuration.json',cfg);print(json.dumps(cfg,indent=2));return
 target=out/'runs';target.mkdir(exist_ok=True)
 if (target/'CONFIGURATION.json').exists():raise RuntimeError('Existing experiment directory: refusing automatic reuse or overwrite.')
 save(target/'CONFIGURATION.json',cfg);(target/'trainer_source.py').write_bytes(source.read_bytes());(target/'runner_source.py').write_bytes(Path(__file__).read_bytes())
 start=time.monotonic();deadline=start+a.hours*3600;tr=splits['train'];va=splits['val'];efmean=float(ey[ei].mean());efsd=float(ey[ei].std()+1e-8)
 models=[]
 for seed in a.seeds:
  # Identical minibatch row sequence across arms, isolated from model RNG.
  rr=np.random.default_rng(seed);schedule=[rr.permutation(tr)[:cfg['ptb_steps_per_epoch']*1024].reshape(-1,1024) for _ in range(a.epochs)]
  for arm in cfg['arms']:
   if time.monotonic()>deadline:raise TimeoutError('Training budget exhausted; incomplete experiment, no inferential summaries.')
   d=target/f'seed{seed}_{arm}';d.mkdir();torch.manual_seed(seed);np.random.seed(seed);net=T.MaskedMultimodalTransformer(dim_ecg=x.shape[1],dim_meta=meta.shape[1],dim_echo=ex.shape[1],d=128,proj_hidden=256,nhead=4,num_layers=2,ff_dim=256,dropout=.1,head_hidden=128);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);er=np.random.default_rng(seed+100000);best=-np.inf;beststate=None;epoch_best=None;history=[]
   for epoch,blocks in enumerate(schedule,1):
    net.train();em=er.permutation(ei);assert len(em)>=len(blocks)*256
    losses=[]
    for j,ix in enumerate(blocks):
     if time.monotonic()>deadline:
      save(d/'INCOMPLETE.json',dict(completed_epochs=epoch-1,reason='wall-time budget',elapsed=time.monotonic()-start));raise TimeoutError('Incomplete experiment: budget reached.')
     xx=torch.from_numpy(x[ix]);mm=torch.from_numpy(meta[ix]);yy=torch.from_numpy(y[ix]);mm=None if torch.rand(())<.5 else mm
     h,logit,_=net(z_ecg=xx,z_meta=mm,mask_prob=.15,slot_perm=True)
     loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,yy)+.5*T.variance_loss(h,gamma=1)+T.covariance_loss(h)
     if arm!='ptb_only':
      idx=em[j*256:(j+1)*256];eh,_,pred=net(z_echo=torch.from_numpy(ex[idx]),mask_prob=.15,slot_perm=True);loss=loss+torch.nn.functional.mse_loss(pred,torch.from_numpy((ey[idx]-efmean)/efsd))
      if arm=='ptb_ef_alignment':
       lm,lc,lmd=T.conditional_align(h,eh,1);loss=loss+.1*min(1,epoch/5)*(lm+lc+lmd)
     opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach()))
    sch.step();auc,apv=T.ptb_val_auc(net,'cpu',x,meta,y,va);r=dict(epoch=epoch,updates=epoch*len(blocks),loss=np.mean(losses),validation_auc=auc,validation_ap=apv);history.append(r)
    if auc>best:best=auc;epoch_best=epoch;beststate=copy.deepcopy(net.state_dict())
    save(d/'history.json',history);print(f'{seed} {arm} epoch {epoch}/{a.epochs} val_auc={auc:.5f} elapsed={time.monotonic()-start:.0f}s',flush=True)
   torch.save(dict(state_dict=beststate,configuration=cfg,arm=arm,seed=seed,selected_epoch=epoch_best,selected_validation_auc=best,ef_mean=efmean,ef_std=efsd),d/'selected_model.pt');models.append((seed,arm,d))
 # All configurations/models selected before any test scoring.
 scores=[];raw=[];cal=[];vprobs=[];ident=[]
 for seed,arm,d in models:
  ck=torch.load(d/'selected_model.pt',map_location='cpu',weights_only=False) # own just-produced checkpoint
  net=T.MaskedMultimodalTransformer(dim_ecg=x.shape[1],dim_meta=meta.shape[1],dim_echo=ex.shape[1],d=128,proj_hidden=256,nhead=4,num_layers=2,ff_dim=256,dropout=.1,head_hidden=128);net.load_state_dict(ck['state_dict']);net.eval();H={}
  with torch.no_grad():
   for sp,idx in splits.items():
    chunks=[]
    for k in range(0,len(idx),1024):
     ii=idx[k:k+1024];h,_,_=net(z_ecg=torch.from_numpy(x[ii]),z_meta=torch.from_numpy(meta[ii]));chunks.append(h.numpy())
    H[sp]=np.concatenate(chunks);np.save(d/f'ptb_latents_{sp}.npy',H[sp])
  H={k:v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-8) for k,v in H.items()};sc=StandardScaler().fit(H['train']);lr=LogisticRegression(max_iter=5000,class_weight='balanced').fit(sc.transform(H['train']),y[tr]);pv=lr.predict_proba(sc.transform(H['val']))[:,1];p=lr.predict_proba(sc.transform(H['test']))[:,1];cl=LogisticRegression().fit(pv[:,None],y[va]);pc=cl.predict_proba(p[:,None])[:,1];yt=y[splits['test']]
  r=dict(seed=seed,arm=arm,AUC=roc_auc_score(yt,p),AP=average_precision_score(yt,p),Brier=brier_score_loss(yt,p),calibrated_Brier=brier_score_loss(yt,pc));scores.append(r);raw.append(p);cal.append(pc);vprobs.append(pv);ident.append([seed,arm]);save(d/'metrics.json',r)
 np.savez_compressed(target/'predictions.npz',seed=np.array([r[0] for r in ident]),arm=np.array([r[1] for r in ident]),raw_test_probability=raw,calibrated_test_probability=cal,raw_validation_probability=vprobs,test_label=y[splits['test']],validation_label=y[va],test_record_index=splits['test'],validation_record_index=va,test_patient_id=patients)
 import csv
 with (target/'per_seed_arm_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=scores[0]);w.writeheader();w.writerows(scores)
 summaries=[]
 for base,other,label in [('ptb_only','ptb_ef','EF_added'),('ptb_ef','ptb_ef_alignment','alignment_added')]:
  for endpoint in ['AUC','AP','Brier','calibrated_Brier']:
   g=np.array([next(r[endpoint] for r in scores if r['seed']==s and r['arm']==other)-next(r[endpoint] for r in scores if r['seed']==s and r['arm']==base) for s in a.seeds]);n=len(g);se=g.std(ddof=1)/np.sqrt(n);half=stats.t.ppf(.975,n-1)*se;summaries.append(dict(contrast=label,endpoint=endpoint,seeds=a.seeds,gains=g.tolist(),mean=g.mean(),sd=g.std(ddof=1),nominal_t_interval=[g.mean()-half,g.mean()+half],p_t=stats.ttest_1samp(g,0).pvalue,target='fixed benchmark procedure mean; unadjusted retrospective comparison; exact t only under iid Gaussian gains'))
 save(target/'paired_summaries.json',summaries);save(target/'COMPLETE.json',dict(elapsed_seconds=time.monotonic()-start,encoder_fits=len(models),no_historical_recovery=True));print('Completed matched three-arm experiment.',flush=True)
if __name__=='__main__':main()
