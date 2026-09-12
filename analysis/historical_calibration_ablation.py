#!/usr/bin/env python3
"""Bounded reanalysis of saved scores; no model fitting or source-data mutations.

Run with Python using explicit output and input arguments.
Differences are aligned minus control for Brier, Full minus comparator for
ablations. Studentized resamples preserve the paired seed gain. The zero-SD
pivot convention matches reproduce_inference_tables.py: T*=0. Percentiles
use NumPy's default linear interpolation. Ablation CIs are descriptive,
unadjusted, Gaussian paired-t intervals, not simultaneous/equivalence claims.
"""
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy import stats

OUT=BRIER=ABLATION=None

def summary(g):
    g = np.asarray(g, dtype=float)
    n = len(g)
    mean = g.mean()
    sd = g.std(ddof=1)
    se = sd / np.sqrt(n)
    half = stats.t.ppf(.975, n-1) * se
    return dict(n=n, mean=float(mean), sd=float(sd), se=float(se),
                t_ci=[float(mean-half), float(mean+half)],
                t_p=float(stats.ttest_1samp(g, 0).pvalue))

def studentized(g, B=200000, seed=7):
    g = np.asarray(g, dtype=float)
    n = len(g)
    bs = g[np.random.default_rng(seed).integers(0, n, (B, n))]
    bsd = bs.std(axis=1, ddof=1)
    T = np.where(bsd > 0, (bs.mean(axis=1)-g.mean()) /
                 np.where(bsd > 0, bsd/np.sqrt(n), 1), 0.0)
    q_hi, q_lo = np.percentile(T, [97.5, 2.5])
    se = g.std(ddof=1)/np.sqrt(n)
    return dict(B=B, rng='numpy.default_rng (PCG64)', seed=seed,
                zero_sd_count=int((bsd == 0).sum()),
                quantile_method='linear', pivot_quantiles=[float(q_lo),float(q_hi)],
                ci=[float(g.mean()-q_hi*se),float(g.mean()-q_lo*se)])

def main():
    global OUT,BRIER,ABLATION
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--brier',required=True);ap.add_argument('--ablation',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=False);BRIER=Path(args.brier);ABLATION=Path(args.ablation)
    r = list(csv.DictReader(BRIER.open()))
    pairs = {}
    for x in r:
        pairs.setdefault(int(x['seed']), {})[x['arm']] = x
    details=[]
    for seed, arms in sorted(pairs.items()):
        z={'seed':seed}
        for scale in ['raw','calibrated']:
            for arm in ['aligned','control']:
                z[f'{scale}_{arm}']=float(arms[arm][f'brier_{scale}'])
            z[f'{scale}_gain']=z[f'{scale}_aligned']-z[f'{scale}_control']
        details.append(z)
    with (OUT/'calibrated_brier_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
    calibrated={}
    for scale in ['raw','calibrated']:
        g=np.array([z[f'{scale}_gain'] for z in details])
        calibrated[scale]=summary(g)
        calibrated[scale]['studentized']=studentized(g)
        calibrated[scale]['positive_count']=int((g>0).sum())
        calibrated[scale]['sign_p']=float(stats.binomtest(int((g>0).sum()),len(g)).pvalue)
    r=list(csv.DictReader(ABLATION.open()))
    full={int(x['seed']):x for x in r if x['config']=='full'}
    ablation=[]
    for cfg in ['full','no_mmd','no_coral','no_mean','no_var','task_only']:
        sub=sorted((x for x in r if x['config']==cfg),key=lambda x:int(x['seed']))
        assert [int(x['seed']) for x in sub]==sorted(full)
        for metric in ['af_auc','af_ap']:
            gains=np.array([float(full[int(x['seed'])][metric])-float(x[metric]) for x in sub])
            z=dict(configuration=cfg,metric=metric,comparison='Full minus configuration',
                   gains=gains.tolist(),comparison_mean=float(np.mean([float(x[metric]) for x in sub])))
            if cfg!='full':z.update(summary(gains))
            else:z.update(n=len(gains),mean=0.0,sd=0.0,se=0.0,t_ci=[0.0,0.0],t_p=None)
            ablation.append(z)
    result=dict(software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BRIER,ABLATION]},
        brier=calibrated,ablation=ablation,
        limitations=['No Brier column exists in saved loss-ablation CSV.',
        'Task-only removes mean/CORAL/MMD and variance penalties; both task losses remain.',
        'New sensitivity analyses do not change existing reported scientific results.',
        'Intervals are retrospective and unadjusted; t validity requires its distributional assumptions.'])
    (OUT/'saved_csv_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
