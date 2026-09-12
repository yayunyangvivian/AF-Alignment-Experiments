import numpy as np,pandas as pd
from scipy import stats
KEYS=['AUC','AP','Brier','calibrated_Brier']
CONTRASTS=[('anchored','none'),('symmetric','none'),('anchored','symmetric')]
def paired(df,conditions,seed_group='combined'):
 rows=[];gains=[]
 iterator=[((),df)] if not conditions else df.groupby(conditions,dropna=False,sort=False)
 for cond,d in iterator:
  cond=(cond,) if len(conditions)==1 and not isinstance(cond,tuple) else cond
  base=dict(zip(conditions,cond))
  for candidate,reference in CONTRASTS:
   a=d[d.arm==candidate].set_index('seed');b=d[d.arm==reference].set_index('seed');ids=sorted(set(a.index)&set(b.index))
   for k in KEYS:
    g=(a.loc[ids,k]-b.loc[ids,k]).to_numpy();sign=1 if k in ['AUC','AP'] else -1
    for s,v in zip(ids,g):gains.append(dict(**base,seed_group=seed_group,contrast=candidate+'-minus-'+reference,seed=s,endpoint=k,difference=float(v)))
    rows.append(dict(**base,seed_group=seed_group,contrast=candidate+'-minus-'+reference,endpoint=k,**summary(g,sign)))
 return pd.DataFrame(rows),pd.DataFrame(gains)

def summary(g,sign=1):
 g=np.asarray(g);n=len(g);sd=float(g.std(ddof=1)) if n>1 else float('nan');mean=float(g.mean());half=stats.t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 and sd>0 else float('nan')
 return dict(K=n,mean=mean,median=float(np.median(g)),sd=sd,lower=float(mean-half),upper=float(mean+half),candidate_favored=int(np.sum(sign*g>0)),reference_favored=int(np.sum(sign*g<0)),ties=int(np.sum(g==0)),interval='nominal paired t mean reference; fixed records; unadjusted; median descriptive')
