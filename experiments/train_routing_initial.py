from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
#!/usr/bin/env python3
"""Fixed-size experiment of an actual v2b alignment-gradient modification."""
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time,datetime,copy,importlib.util,csv,argparse,platform
import numpy as np
import torch,sklearn,scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from scipy import stats
from af_alignment.probe import validation_probe_auc,export_latents,fit_probe,predict_probe
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 global ROOT
 ap=argparse.ArgumentParser();ap.add_argument('--ptb-dir',required=True);ap.add_argument('--echo-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
 from af_alignment.runtime import ROOT as REPO,check_inputs,output_dir
 lock=json.loads((REPO/'configs/routing_initial.json').read_text())
 base=json.loads((REPO/'configs/training_common.json').read_text());base['inputs']=check_inputs(Path(a.ptb_dir),Path(a.echo_dir))
 if not a.run:print('Retained input hashes verified; no training.');return
 ROOT=output_dir(a.output);out=ROOT/'runs';out.mkdir()
 from af_alignment import v2b as T
 lock.update(source_sha256=sha(T.__file__),function_sha256=sha(REPO/'src/af_alignment/probe.py'),runner_sha256=sha(__file__))
 cfg={k:base[k] for k in ['dimension','proj_hidden','nhead','layers','ff_dim','head_hidden','dropout','ptb_batch','echo_batch','ptb_steps_per_epoch','lr','weight_decay','grad_clip','ptb_positive_weight','mask_probability','metadata_dropout','slot_permutation','EF_standardization','variance_weight','covariance_weight','variance_gamma','warmup_epochs','probe','inputs','variance_penalty','covariance_penalty']}
 cfg.update(protocol=lock,seeds=lock['seeds'],epochs=30,arms=lock['arms'],alignment_weight_by_arm=lock['alignment_weight_by_arm'],regularization_cohort='PTB only in all three arms',cohort_use='All three arms use identical PTB AF and Echo EF supervision; only alignment weight or gradient routing differs',selection='maximum validation frozen-probe AUC, earliest epoch on ties, in all three arms',pairing='identical per-seed initialization and PTB/Echo schedules; alignment computations in all three arms preserve common stochastic draw consumption',source='src/af_alignment/v2b.py',source_sha256=lock['source_sha256'],function_sha256=lock['function_sha256'],runner_sha256=lock['runner_sha256'],max_hours=lock['budget_seconds']/3600,threads=2,environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform()))
 save(out/'CONFIGURATION.json',cfg);save(ROOT/'STATUS.json',dict(stage='training',utc=utc()))
 torch.set_num_threads(2)
 ptb=Path(a.ptb_dir);echo=Path(a.echo_dir);x=np.load(ptb/'emb_ecg.npy').astype('float32');meta=np.load(ptb/'tab.npy').astype('float32');y=np.load(ptb/'y_af.npy').astype('float32');tr=np.loadtxt(ptb/'ids_train.txt',dtype=int);va=np.loadtxt(ptb/'ids_val.txt',dtype=int)
 ex=np.load(echo/'emb_echo.npy').astype('float32');ey=np.load(echo/'y_ef.npy').astype('float32');ei=np.loadtxt(echo/'ids_echo_train.txt',dtype=int);efmean=float(ey[ei].mean());efsd=float(ey[ei].std()+1e-8)
 assert len(ei)>=17*256 and len(tr)//1024==17
 def new_net():return T.MaskedMultimodalTransformer(dim_ecg=x.shape[1],dim_meta=meta.shape[1],dim_echo=512,d=128,proj_hidden=256,nhead=4,num_layers=2,ff_dim=256,dropout=.1,head_hidden=128)
 start=time.monotonic();deadline=start+lock['budget_seconds'];selected=[]
 for seed in lock['seeds']:
  rr=np.random.default_rng(seed);schedule=np.stack([rr.permutation(tr)[:17*1024].reshape(-1,1024) for _ in range(30)]);schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
  for arm in lock['arms']:
   d=out/f'seed{seed}_{arm}';d.mkdir();torch.manual_seed(seed);np.random.seed(seed);net=new_net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);er=np.random.default_rng(seed+100000);history=[];best=-np.inf;weight=lock['alignment_weight_by_arm'][arm]
   initial_sha=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
   save(d/'PAIRING.json',dict(seed=seed,arm=arm,initial_state_sha256=initial_sha,ptb_schedule_sha256=schedule_sha))
   for epoch,blocks in enumerate(schedule,1):
    net.train();em=er.permutation(ei);epoch_losses=[]
    for j,ix in enumerate(blocks):
     if time.monotonic()>deadline:raise TimeoutError('Fixed budget exceeded. No complete-results claim or replacement seed.')
     xx=torch.from_numpy(x[ix]);mm=torch.from_numpy(meta[ix]);yy=torch.from_numpy(y[ix]);mm=None if torch.rand(())<.5 else mm
     h,logit,_=net(z_ecg=xx,z_meta=mm,mask_prob=.15,slot_perm=True)
     afloss=torch.nn.functional.binary_cross_entropy_with_logits(logit,yy);reg=.5*T.variance_loss(h,gamma=1)+T.covariance_loss(h)
     ixef=em[j*256:(j+1)*256];eh,_,pred=net(z_echo=torch.from_numpy(ex[ixef]),mask_prob=.15,slot_perm=True);efloss=torch.nn.functional.mse_loss(pred,torch.from_numpy((ey[ixef]-efmean)/efsd))
     # Same forward discrepancy and bandwidth draws; candidate changes its PTB gradient.
     if arm=='anchored':lm,lc,lmd=T.af_anchored_alignment(h,eh)
     else:lm,lc,lmd=T.conditional_align(h,eh,1)
     alignment=lm+lc+lmd
     loss=afloss+reg+efloss+weight*min(1,epoch/5)*alignment
     opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
     epoch_losses.append([float(q.detach()) for q in [afloss,efloss,reg,alignment,loss]])
    scheduler.step();head_auc,head_ap=T.ptb_val_auc(net,'cpu',x,meta,y,va);probe_auc=validation_probe_auc(net,x,meta,tr,va,y[tr],y[va]);means=np.mean(epoch_losses,axis=0)
    record=dict(epoch=epoch,updates=epoch*17,af_loss=means[0],ef_loss=means[1],regularization_loss=means[2],unweighted_alignment_loss=means[3],total_loss=means[4],validation_head_auc=head_auc,validation_head_ap=head_ap,validation_probe_auc=probe_auc,torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),echo_schedule_sha256=hashlib.sha256(em.tobytes()).hexdigest(),utc=utc());history.append(record)
    if probe_auc>best:
     best=probe_auc;torch.save(dict(state_dict=copy.deepcopy(net.state_dict()),seed=seed,arm=arm,selected_epoch=epoch,selected_validation_probe_auc=best,selected_validation_head_auc=float(head_auc),ef_mean=efmean,ef_std=efsd,configuration=cfg),d/'selected_model.pt')
    save(d/'history.json',history);print(f'{seed} {arm} epoch {epoch}/30 val_probe={probe_auc:.5f} elapsed={time.monotonic()-start:.0f}s',flush=True)
   selected.append((seed,arm,d))
 save(ROOT/'SELECTION_COMPLETE.json',dict(utc=utc(),seed_triples=5,encoder_fits=15));save(ROOT/'STATUS.json',dict(stage='evaluation',utc=utc()))
 te=np.loadtxt(ptb/'ids_test.txt',dtype=int);patient=np.load(ptb/'test_patient_ids.npy');rows=[];raw=[];calibrated=[];vprobs=[]
 for seed,arm,d in selected:
  ck=torch.load(d/'selected_model.pt',map_location='cpu',weights_only=False);net=new_net();net.load_state_dict(ck['state_dict']);net.eval();H={sp:export_latents(net,x,meta,ii) for sp,ii in [('train',tr),('val',va),('test',te)]}
  for sp,h in H.items():np.save(d/f'latents_{sp}.npy',h)
  sc,lr=fit_probe(H['train'],y[tr]);pv=predict_probe(sc,lr,H['val']);pt=predict_probe(sc,lr,H['test']);cal=LogisticRegression().fit(pv[:,None],y[va]);pc=cal.predict_proba(pt[:,None])[:,1]
  np.savez_compressed(d/'probe_parameters.npz',scaler_mean=sc.mean_,scaler_scale=sc.scale_,probe_coef=lr.coef_,probe_intercept=lr.intercept_,classes=lr.classes_,calibrator_coef=cal.coef_,calibrator_intercept=cal.intercept_)
  row=dict(seed=seed,arm=arm,selected_epoch=ck['selected_epoch'],validation_probe_auc=roc_auc_score(y[va],pv),AUC=roc_auc_score(y[te],pt),AP=average_precision_score(y[te],pt),Brier=brier_score_loss(y[te],pt),calibrated_Brier=brier_score_loss(y[te],pc));rows.append(row);raw.append(pt);calibrated.append(pc);vprobs.append(pv)
 with (out/'per_seed_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'predictions.npz',seed=np.array([row['seed'] for row in rows]),arm=np.array([row['arm'] for row in rows]),raw_test_probability=raw,calibrated_test_probability=calibrated,raw_validation_probability=vprobs,test_label=y[te],validation_label=y[va],test_record_index=te,validation_record_index=va,test_patient_id=patient)
 summaries=[]
 for reference,contrast in [('symmetric','anchored_vs_original'),('none','anchored_vs_none')]:
  for endpoint in lock['endpoints']:
   g=np.array([next(row[endpoint] for row in rows if row['seed']==seed and row['arm']=='anchored')-next(row[endpoint] for row in rows if row['seed']==seed and row['arm']==reference) for seed in lock['seeds']]);sd=float(g.std(ddof=1));half=stats.t.ppf(.975,4)*sd/np.sqrt(5)
   summaries.append(dict(contrast=contrast,endpoint=endpoint,gains=g.tolist(),mean=float(g.mean()),sd=sd,nominal_t_interval=[float(g.mean()-half),float(g.mean()+half)],p_t=float(stats.ttest_1samp(g,0).pvalue) if sd else None))
 save(out/'summary.json',summaries);save(out/'COMPLETE.json',dict(utc=utc(),seed_triples=5,encoder_fits=15,elapsed_seconds=time.monotonic()-start));save(ROOT/'STATUS.json',dict(stage='evaluation_complete_verification_pending',utc=utc()));print('All15 fits and eight endpoint comparisons complete; verification pending.',flush=True)
if __name__=='__main__':main()
