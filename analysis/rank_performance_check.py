#!/usr/bin/env python3
"""Join saved per-seed AUC to covariance ranks of the exact exported latents.

No fitting. The configuration paths are those in reproduce_inference_tables.py.
Recomputed aligned ranks must match the existing per-seed CSV before plotting.
Covariance-pair head scores are descriptive: no verified one-factor training
manifest is available, and these heads are not the primary logistic probes.
"""
import csv, hashlib, json, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/plot_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'outputs/cache'))
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score,brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=None
AF=None

def rank(path):
    z=np.asarray(np.load(path),np.float64)
    z=z-z.mean(axis=0)
    cov=z.T@z/(len(z)-1)
    lam=np.clip(np.linalg.eigvalsh(cov),0,None)
    return float(lam.sum()**2/np.dot(lam,lam))

def main():
    global AF,OUT
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True,help='input archive containing data/ptbxl, data/seed_runs and historical score directories');ap.add_argument('--output',required=True);a=ap.parse_args();AF=Path(a.archive);OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=False)
    records=[]
    configs=[(1,'results_encoder_multiseed','v2b_seed{}','ptbonly_seed{}'),
             (2,'results_encoder_multiseed_fix3_clean','v2b_fixed_c4b9fa0_seed{}','matched_fixed_c4b9fa0_seed{}')]
    for cfg,dirname,apath,cpath in configs:
        csvpath=AF/'data'/dirname/'per_seed.csv'
        rows=list(csv.DictReader(csvpath.open()))
        for row in rows:
            seed=int(row['seed'])
            for arm,stem,metric in [('aligned',apath,'v2b_AUC'),('control',cpath,'ptbonly_AUC')]:
                latent=AF/'data/seed_runs'/stem.format(seed)/'ptbxl_h_test.npy'
                pr=rank(latent)
                if arm=='aligned':assert np.isclose(pr,float(row['v2b_effrank_PR']),rtol=1e-10)
                records.append(dict(configuration=cfg,seed=seed,arm=arm,auc=float(row[metric]),
                    effective_rank=pr,latent_path=str(latent),latent_sha256=hashlib.sha256(latent.read_bytes()).hexdigest(),
                    auc_csv=str(csvpath),auc_csv_sha256=hashlib.sha256(csvpath.read_bytes()).hexdigest()))
    with (OUT/'rank_performance_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
    for cfg,ax in enumerate(axes,1):
        for arm,color,marker in [('aligned','#0072B2','o'),('control','#D55E00','s')]:
            r=[z for z in records if z['configuration']==cfg and z['arm']==arm]
            ax.scatter([z['effective_rank'] for z in r],[z['auc'] for z in r],label=arm.capitalize(),color=color,marker=marker,s=30)
        ax.set_xlabel(f'Configuration {cfg}: effective rank')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Frozen logistic-probe test AUC')
    axes[1].legend(frameon=False,fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/'rank_performance.pdf',bbox_inches='tight')
    fig.savefig(OUT/'rank_performance.png',dpi=180,bbox_inches='tight')
    variants=[]
    for variant in ['aligned_h_coral','aligned_h_coral_nocollapse']:
        d=AF/'data'/variant
        y=np.load(d/'ptbxl_y_test.npy').ravel()
        logit=np.load(d/'ptbxl_logit_test.npy').ravel()
        p=expit(logit)
        variants.append(dict(variant=variant,n=len(y),effective_rank=rank(d/'ptbxl_h_test.npy'),
              end_to_end_auc=roc_auc_score(y,p),end_to_end_ap=average_precision_score(y,p),raw_brier=brier_score_loss(y,p),
              prediction_sha256=hashlib.sha256((d/'ptbxl_logit_test.npy').read_bytes()).hexdigest()))
    result=dict(covariance_named_variants=variants,rank_plot_records=len(records),
      rank_verification='Every aligned recomputed rank matches saved per-seed CSV at rtol=1e-10.',
      limitations=['Rank is from raw held-out latent covariance; AUC uses the existing row-normalized frozen logistic probe.',
      'Scatter is descriptive across training runs, not a causal test or mechanism identification.',
      'The two named covariance variants have saved end-to-end heads; a one-factor training manifest is unverified.',
      'Named covariance comparisons do not isolate one repair factor.'])
    (OUT/'rank_performance_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
