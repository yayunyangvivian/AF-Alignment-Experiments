from common import *
from scipy.special import logit

def main():
 z=np.load(B/'clean_predictions.npz');y=z['label'].astype(float);pi=y.mean();prior=pd.read_csv(B/'calibration_per_seed.csv');rows=[];bins=[]
 for j,(seed,arm) in enumerate(zip(z['seed'],z['arm'])):
  if arm not in ARMS:continue
  for kind,key in [('raw','raw_probability'),('PTB_calibrated','calibrated_probability')]:
   p=z[key][j].astype(float);mean=p.mean();var=p.var();bias=(mean-pi)**2;cov=np.mean((p-mean)*(y-pi));bs=np.mean((p-y)**2);identity=var+bias+pi*(1-pi)-2*cov;assert abs(identity-bs)<1e-12
   old=prior[(prior.seed==seed)&(prior.arm==arm)&(prior.kind==kind)].iloc[0];assert abs(old.Brier-bs)<1e-10
   q=np.quantile(p,[.01,.1,.25,.5,.75,.9,.99]);l=logit(np.clip(p,1e-6,1-1e-6));
   rows.append(dict(seed=seed,arm=arm,kind=kind,mean_probability=mean,sd_probability=p.std(),mean_logit=l.mean(),sd_logit=l.std(),clipped=int(np.sum((p<1e-6)|(p>1-1e-6))),class0_mean=p[y==0].mean(),class1_mean=p[y==1].mean(),class0_mse=np.mean(p[y==0]**2),class1_mse=np.mean((1-p[y==1])**2),probability_variance=var,squared_mean_bias=bias,negative_twice_covariance=-2*cov,outcome_variance=pi*(1-pi),Brier=bs,calibration_in_large=old.calibration_in_large,intercept=old.intercept,slope=old.slope,ECE10=old.ECE10,**dict(zip(['q01','q10','q25','q50','q75','q90','q99'],q))))
   for cls in [0,1]:
    counts,edges=np.histogram(p[y==cls],bins=np.linspace(0,1,31))
    for i,n in enumerate(counts):bins.append(dict(seed=seed,arm=arm,kind=kind,outcome=cls,bin=i,midpoint=(edges[i]+edges[i+1])/2,fraction=n/(y==cls).sum()))
 d=table('calibration_probability_per_seed.csv',rows);table('probability_histograms.csv',bins);s=[];gains=[]
 fields=['calibration_in_large','intercept','slope','ECE10','Brier','mean_probability','sd_probability','class0_mean','class1_mean','probability_variance','squared_mean_bias','negative_twice_covariance']
 for kind,v in d.groupby('kind'):
  for cand,ref in PAIRS:
   a=v[v.arm==cand].set_index('seed');b=v[v.arm==ref].set_index('seed')
   for k in fields:
    g=a[k]-b[k];s.append(dict(kind=kind,contrast=cand+'-minus-'+ref,quantity=k,**summary(g)))
    for seed,h in g.items():gains.append(dict(kind=kind,contrast=cand+'-minus-'+ref,quantity=k,seed=seed,difference=h))
 table('calibration_paired_summary.csv',s);table('calibration_seed_contrasts.csv',gains)
 table('calibration_arm_summary.csv',d.groupby(['arm','kind'])[fields].mean().reset_index())
 f,axs=plt.subplots(2,4,figsize=(13,6))
 for i,kind in enumerate(['raw','PTB_calibrated']):
  for j,k in enumerate(['mean_probability','slope','ECE10','Brier']):
   ax=axs[i,j]
   for seed in SEEDS:
    a=d[(d.seed==seed)&(d.kind==kind)].set_index('arm');ax.plot(range(3),a.loc[ARMS,k],color='grey',alpha=.35,lw=.6)
   for n,arm in enumerate(ARMS):
    a=d[(d.arm==arm)&(d.kind==kind)];ax.scatter(np.full(10,n),a[k],color=COLORS[arm],s=15)
   ax.set_title(kind+': '+k);ax.set_xticks(range(3),ARMS,rotation=20)
   if k=='slope':ax.axhline(1,color='black',ls=':',lw=.6)
   if k=='mean_probability':ax.axhline(pi,color='black',ls=':',lw=.6)
 f.suptitle('Frozen probabilities and diagnostic calibration; lines connect training seeds');f.tight_layout();fig('calibration_probability_audit',f)
 b=pd.DataFrame(bins);f,axs=plt.subplots(2,2,figsize=(10,7))
 for i,kind in enumerate(['raw','PTB_calibrated']):
  for cls in [0,1]:
   ax=axs[i,cls]
   for arm in ARMS:
    v=b[(b.arm==arm)&(b.kind==kind)&(b.outcome==cls)].groupby('midpoint').fraction.mean();ax.plot(v.index,v.values,color=COLORS[arm],label=arm)
   ax.set_title(f'{kind}; outcome={cls}');ax.set_xlabel('Frozen probability');ax.set_ylabel('Mean within-class bin fraction')
 axs[0,0].legend();f.tight_layout();fig('probability_distributions',f)
 save('CALIBRATION_AUDIT_COMPLETE.json',dict(models=30,records=len(y),new_Georgia_calibration_fits=0,identity_max_error='asserted <1e-12 for all60 model/kind rows'))
 print('Calibration audit complete; existing diagnostics reused, frozen predictions unchanged.')
if __name__=='__main__':main()
