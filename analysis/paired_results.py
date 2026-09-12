"""Recompute fixed-benchmark paired summaries from retained seed scores."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from scipy import stats
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scores',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);df=pd.read_csv(a.scores)
    groups=[c for c in ['condition','subgroup','dose'] if c in df]
    rows=[];gains=[]
    for values,d in (df.groupby(groups,dropna=False) if groups else [((),df)]):
        if not isinstance(values,tuple):values=(values,)
        extra=dict(zip(groups,values));x=d[d.arm==a.candidate].set_index('seed');y=d[d.arm==a.reference].set_index('seed')
        assert x.index.is_unique and y.index.is_unique and set(x.index)==set(y.index) and len(x)>=2,'Incomplete or duplicated pairing'
        ids=sorted(x.index)
        for k in ['AUC','AP','Brier','calibrated_Brier']:
            g=(x.loc[ids,k]-y.loc[ids,k]).to_numpy();n=len(g);sd=g.std(ddof=1);half=stats.t.ppf(.975,n-1)*sd/np.sqrt(n);m=g.mean();direction=1 if k in ['AUC','AP'] else -1
            rows.append(dict(**extra,endpoint=k,K=n,mean=m,median=np.median(g),sd=sd,lower=m-half,upper=m+half,candidate_favored=int((direction*g>0).sum()),reference_favored=int((direction*g<0).sum()),ties=int((g==0).sum()),scope='nominal paired-t mean interval; fixed evaluation sample; retrospective; unadjusted'))
            gains.extend(dict(**extra,seed=s,endpoint=k,difference=v) for s,v in zip(ids,g))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(gains).to_csv(out/'seed_gains.csv',index=False)
if __name__=='__main__':main()
