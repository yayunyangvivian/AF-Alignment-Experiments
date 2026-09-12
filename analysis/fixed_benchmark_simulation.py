from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
#!/usr/bin/env python3
"""New fixed-benchmark sensitivity grid; historical simulations untouched.

Independent total block errors G_j=u_j, mean zero and variance one. There is
no random patient/benchmark effect in this conditional experiment. Laws:
N(0,1); Exp(1)-1; t_3/sqrt(3); mixture (0.9*N(0,1)+0.1*N(0,9))/sqrt(1.8),
where the normal's second parameter is variance and the mixture selects one
component per block. Scale/translation-equivariance permits unit variance.
All intervals below nominal 95%; rejection of the corresponding zero-mean
null is exactly noncoverage of zero, evaluated on the same simulations.
The sign interval instead covers the true median, log(2)-1 for Exp(1)-1 and
zero otherwise. Its equal-tail nonrandomized interval uses the largest
order index k with 2*P(Bin(K,.5)<=k-1)<=.05. At k=0 it is (-inf,+inf).

Studentization uses 999 index resamples of each K-vector, zero pivot when
all resampled observations are identical (exact zero resample variance),
and empirical linear-interpolation quantiles with ordinary inversion.
Resample indices are shared across laws within a K but independent of data;
this coupling reduces computational cost without changing cell marginals.
The procedure remains a finite numerical experiment, not a guarantee for
arbitrary skew/tails or a population-uncertainty calculation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from scipy import stats

import argparse
OUT=None
M=5000; B=999; CHUNK=100; SEED=20260906
LAWS=['normal','shifted_exponential','t3','contamination']

def draw(rng,law,shape):
    if law=='normal':return rng.normal(size=shape)
    if law=='shifted_exponential':return rng.exponential(size=shape)-1
    if law=='t3':return rng.standard_t(3,size=shape)/np.sqrt(3)
    z=rng.normal(size=shape)
    return z*np.where(rng.random(shape)<.1,3.,1.)/np.sqrt(1.8)

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    from af_alignment.runtime import output_dir
    OUT=output_dir(a.output)
    start=time.monotonic();rows=[];validation=[]
    for ki,K in enumerate([5,10,20]):
        valid=[k for k in range(1,K//2+1) if 2*stats.binom.cdf(k-1,K,.5)<=.05]
        k=max(valid,default=0)
        sign_exact=1. if k==0 else float(1-2*stats.binom.cdf(k-1,K,.5))
        for li,law in enumerate(LAWS):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,ki,li,0]))
            resampler=np.random.default_rng(np.random.SeedSequence([SEED,ki,1]))
            med=np.log(2)-1 if law=='shifted_exponential' else 0.
            counts=dict(paired_t=0,studentized=0,median_sign=0)
            zero_count=0
            for j in range(0,M,CHUNK):
                n=min(CHUNK,M-j);g=draw(rng,law,(n,K));mean=g.mean(1);se=g.std(1,ddof=1)/np.sqrt(K)
                half=stats.t.ppf(.975,K-1)*se
                counts['paired_t']+=int(((mean-half<=0)&(mean+half>=0)).sum())
                idx=resampler.integers(0,K,(n,B,K))
                bs=np.take_along_axis(g[:,None,:],idx,axis=2)
                zero=(idx==idx[:,:,:1]).all(2)
                bse=bs.std(2,ddof=1)/np.sqrt(K)
                T=(bs.mean(2)-mean[:,None])/np.where(zero,1.,bse)
                T[zero]=0.;zero_count+=int(zero.sum())
                ql,qh=np.quantile(T,[.025,.975],axis=1,method='linear')
                lo=mean-qh*se;hi=mean-ql*se
                counts['studentized']+=int(((lo<=0)&(hi>=0)).sum())
                if k==0:counts['median_sign']+=n
                else:
                    ordered=np.sort(g,axis=1)
                    counts['median_sign']+=int(((ordered[:,k-1]<=med)&(ordered[:,K-k]>=med)).sum())
                if time.monotonic()-start>290:raise RuntimeError('Five-minute runtime budget approached.')
            for method,count in counts.items():
                p=count/M
                row=dict(law=law,K=K,method=method,target='median' if method=='median_sign' else 'mean',
                    n_sim=M,bootstrap_resamples=B if method=='studentized' else 0,covered_count=count,
                    coverage_pct=100*p,coverage_mcse_pp=100*np.sqrt(p*(1-p)/M),
                    rejection_count=M-count,rejection_pct=100*(1-p),
                    true_mean=0.,true_median=float(med),sign_order_index=k if method=='median_sign' else '',
                    zero_variance_bootstraps=zero_count if method=='studentized' else '',
                    exact_coverage_pct=100*sign_exact if method=='median_sign' else (95. if method=='paired_t' and law=='normal' else ''))
                rows.append(row)
                if method=='median_sign' or (method=='paired_t' and law=='normal'):
                    exact=sign_exact if method=='median_sign' else .95
                    sem=np.sqrt(exact*(1-exact)/M)
                    z=0. if sem==0 and p==exact else (p-exact)/sem
                    validation.append(dict(law=law,K=K,method=method,coverage=p,exact=exact,
                                           exact_mcse=sem,z_difference=float(z),within_four_mcse=bool(abs(z)<=4)))
            print(f'{law} K={K}: '+str(counts)+f' elapsed={time.monotonic()-start:.1f}s',flush=True)
    with (OUT/'fixed_benchmark_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    metadata=dict(methodology=__doc__,M=M,B=B,chunk=CHUNK,seed=SEED,validation=validation,
        software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-start)
    (OUT/'fixed_benchmark_sensitivity_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert all(x['within_four_mcse'] for x in validation), 'Analytic calibration exceeds four MCSE; inspect.'
    print('Analytic checks:',json.dumps(validation,indent=2))

if __name__=='__main__':main()
