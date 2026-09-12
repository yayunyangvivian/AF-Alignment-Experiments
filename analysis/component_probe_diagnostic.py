#!/usr/bin/env python3
"""Exploratory training-only PCA probe of frozen raw encoder latents.

Encoder training and primary evaluation are untouched. For each of the 30
existing fitted encoders, fit full PCA on centered raw training latents; apply
that rotation to test. Fit three class-balanced L2 logistic probes (C=1,
lbfgs, max_iter=5000), each with StandardScaler fit on training coordinates:
PC1 only, every remaining PC, all PCs. Do NOT row-normalize projected data.
These probes are distinct from the primary row-normalized representation
probe. Report descriptive per-arm seed means and SDs, not causal effects.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

AF=None
OUT=None

def main():
    global AF,OUT
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True,help='input archive containing data/ptbxl, data/seed_runs and historical score directories');ap.add_argument('--output',required=True);a=ap.parse_args();AF=Path(a.archive);OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    ptb=AF/'data/ptbxl'
    y=np.load(ptb/'y_af.npy').astype(float)
    ytr=y[np.loadtxt(ptb/'ids_train.txt',dtype=int).ravel()]
    yte=y[np.loadtxt(ptb/'ids_test.txt',dtype=int).ravel()]
    records=[]
    configs=[(1,range(42,47),'v2b_seed{}','ptbonly_seed{}'),
             (2,range(42,52),'v2b_fixed_c4b9fa0_seed{}','matched_fixed_c4b9fa0_seed{}')]
    for cfg,seeds,apath,cpath in configs:
        for seed in seeds:
            for arm,stem in [('aligned',apath),('control',cpath)]:
                d=AF/'data/seed_runs'/stem.format(seed)
                trfile=d/'ptbxl_h_train.npy';tefile=d/'ptbxl_h_test.npy'
                tr=np.asarray(np.load(trfile),np.float64);te=np.asarray(np.load(tefile),np.float64)
                assert len(tr)==len(ytr) and len(te)==len(yte)
                pca=PCA(svd_solver='full').fit(tr)
                trp=pca.transform(tr);tep=pca.transform(te)
                for mode,cols in [('PC1',slice(0,1)),('remaining',slice(1,None)),('all',slice(None))]:
                    sc=StandardScaler().fit(trp[:,cols]);xtr=sc.transform(trp[:,cols]);xte=sc.transform(tep[:,cols])
                    with warnings.catch_warnings(record=True) as observed:
                        warnings.simplefilter('always')
                        lr=LogisticRegression(class_weight='balanced',C=1,solver='lbfgs',max_iter=5000).fit(xtr,ytr)
                    prob=lr.predict_proba(xte)[:,1]
                    records.append(dict(configuration=cfg,seed=seed,arm=arm,probe=mode,
                        dimensions=xtr.shape[1],auc=roc_auc_score(yte,prob),ap=average_precision_score(yte,prob),
                        raw_brier=brier_score_loss(yte,prob),training_pc1_variance_fraction=float(pca.explained_variance_ratio_[0]),
                        iterations=int(lr.n_iter_[0]),warnings=' | '.join(str(x.message) for x in observed),
                        latent_directory=str(d),train_sha256=hashlib.sha256(trfile.read_bytes()).hexdigest(),
                        test_sha256=hashlib.sha256(tefile.read_bytes()).hexdigest()))
                print(f'Finished configuration={cfg} seed={seed} arm={arm}; elapsed={time.monotonic()-started:.1f}s',flush=True)
                if time.monotonic()-started>290:
                    raise RuntimeError('Bounded five-minute resource limit approached; no complete summary emitted.')
    with (OUT/'component_probe_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    summaries=[]
    for cfg in (1,2):
        for arm in ('aligned','control'):
            for mode in ('PC1','remaining','all'):
                rows=[r for r in records if r['configuration']==cfg and r['arm']==arm and r['probe']==mode]
                s=dict(configuration=cfg,arm=arm,probe=mode,n=len(rows))
                for metric in ('auc','ap','raw_brier','training_pc1_variance_fraction'):
                    values=np.array([r[metric] for r in rows]);s[metric+'_mean']=values.mean();s[metric+'_sd']=values.std(ddof=1)
                summaries.append(s)
    result=dict(summary=summaries,elapsed_seconds=time.monotonic()-started,warnings=[r for r in records if r['warnings']],
        methodology=__doc__)
    (OUT/'component_probe_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Component probe numerical outputs saved.')

if __name__=='__main__':main()
