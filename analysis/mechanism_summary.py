from common import *
import argparse

def summarize(include_dose):
 tags=['base','dose03'] if include_dose else ['base']
 def read(prefix):return pd.concat([pd.read_csv(R/f'{prefix}_{tag}.csv') for tag in tags],ignore_index=True)
 rep=read('representation_stats');mov=read('representation_movement');al=read('alignment_diagnostics');metrics=read('forward_metrics');clean=metrics[metrics.condition=='clean'];r=rep[rep.condition=='clean'].merge(clean,on=['seed','arm','dose','condition']).merge(al,on=['seed','arm','dose'],suffixes=('','_alignment'))
 table('mechanism_clean_per_seed.csv',r);table('mechanism_arm_summary.csv',r.groupby(['dose','arm']).mean(numeric_only=True).reset_index())
 sd=[];gains=[];fields=['PR','trace','PC1_share','separation_ratio','probe_separation_ratio','MMD_adaptive','MMD_common']+KEYS
 for dose in [.1]+([.3] if include_dose else []):
  for cand,ref in PAIRS:
   a=r[(r.arm==cand)&(r.dose==dose)].set_index('seed');b=r[(r.arm==ref)&(r.dose==(0 if ref=='none' else dose))].set_index('seed')
   for field in fields:
    g=a.loc[SEEDS,field]-b.loc[SEEDS,field];sd.append(dict(dose=dose,contrast=cand+'-minus-'+ref,quantity=field,**summary(g)))
    for seed,v in g.items():gains.append(dict(dose=dose,contrast=cand+'-minus-'+ref,quantity=field,seed=seed,difference=v))
 table('mechanism_paired_summary.csv',sd);table('mechanism_paired_seed_effects.csv',gains)
 baseline=metrics[metrics.condition=='clean'][['seed','arm','dose']+KEYS];v=metrics[metrics.condition!='clean'].merge(baseline,on=['seed','arm','dose'],suffixes=('','_clean')).merge(mov,on=['seed','arm','dose','condition'])
 for k in KEYS:v[k+'_loss']=-sign(k)*(v[k]-v[k+'_clean'])
 v=v.merge(r[['seed','arm','dose','MMD_common','separation_ratio']],on=['seed','arm','dose'],validate='many_to_one')
 table('movement_performance_per_seed.csv',v);ms=[]
 for (dose,condition),group in v.groupby(['dose','condition']):
  if dose==0:continue
  for cand,ref in PAIRS:
   a=v[(v.arm==cand)&(v.dose==dose)&(v.condition==condition)].set_index('seed');b=v[(v.arm==ref)&(v.dose==(0 if ref=='none' else dose))&(v.condition==condition)].set_index('seed')
   for field in ['rms_relative','mean_distance','mean_cosine_distance','probe_rms_relative']+[k+'_loss' for k in KEYS]+KEYS:
    g=a.loc[SEEDS,field]-b.loc[SEEDS,field];ms.append(dict(dose=dose,condition=condition,contrast=cand+'-minus-'+ref,quantity=field,**summary(g)))
 table('movement_stress_paired_summary.csv',ms)
 cs=[];scatter=[]
 def add(kind,arm,dose,condition,xname,yname,x,y):
  cs.append(dict(kind=kind,arm=arm,dose=dose,condition=condition,x=xname,y=yname,K=len(x),**corr(x,y)))
  for seed,xx,yy in zip(SEEDS,x,y):scatter.append(dict(kind=kind,arm=arm,dose=dose,condition=condition,x=xname,y=yname,seed=seed,x_value=xx,y_value=yy))
 for (dose,arm),a in r.groupby(['dose','arm']):
  a=a.sort_values('seed')
  for xname in ['MMD_common','MMD_adaptive','PR','trace','separation_ratio','probe_separation_ratio']:
   for k in ['AUC','AP']:add('within_arm',arm,dose,'clean',xname,k,a[xname],a[k])
 for (dose,arm,cond),a in v.groupby(['dose','arm','condition']):
  a=a.sort_values('seed')
  for xname in ['rms_relative','probe_rms_relative']:
   for k in ['AUC','AP']:
    add('within_arm',arm,dose,cond,xname,k+'_loss',a[xname],a[k+'_loss'])
    add('within_arm',arm,dose,cond,xname,k+'_clean',a[xname],a[k+'_clean'])
  add('within_arm',arm,dose,cond,'MMD_common','rms_relative',a.MMD_common,a.rms_relative)
 for dose in [.1]+([.3] if include_dose else []):
  a=r[(r.arm=='anchored')&(r.dose==dose)].set_index('seed').loc[SEEDS];b=r[(r.arm=='symmetric')&(r.dose==dose)].set_index('seed').loc[SEEDS]
  for xname in ['MMD_common','PR','separation_ratio','probe_separation_ratio']:
   for k in ['AUC','AP']:add('paired_change','anchored-minus-symmetric',dose,'clean',xname,k,a[xname]-b[xname],a[k]-b[k])
 table('mechanism_correlations.csv',cs);table('mechanism_correlation_points.csv',scatter)
 # Main mechanism overview: all seed dots, no patient-level inference.
 fields0=['PR','trace','PC1_share','separation_ratio','probe_separation_ratio','MMD_common']
 f,axs=plt.subplots(2,3,figsize=(12,7))
 for ax,k in zip(axs.ravel(),fields0):
  for seed in SEEDS:
   a=r[(r.seed==seed)&r.dose.isin([0,.1])].set_index('arm');ax.plot(range(3),a.loc[ARMS,k],c='grey',alpha=.3,lw=.7)
  for j,arm in enumerate(ARMS):ax.scatter(np.full(10,j),r[(r.arm==arm)&r.dose.isin([0,.1])][k],color=COLORS[arm],s=18)
  ax.set_title(k);ax.set_xticks(range(3),ARMS,rotation=20)
 f.suptitle('Representation diagnostics: each dot is a training seed; baseline dose0/.1');f.tight_layout();fig('representation_mechanism',f)
 f,axs=plt.subplots(2,4,figsize=(14,7));conds=list(v.condition.unique())
 for ax,cond in zip(axs.ravel(),conds):
  for j,arm in enumerate(ARMS):
   a=v[(v.arm==arm)&v.dose.isin([0,.1])&(v.condition==cond)];ax.scatter(np.full(10,j),a.rms_relative,color=COLORS[arm],s=15)
  ax.set_title(cond);ax.set_xticks(range(3),ARMS,rotation=20);ax.set_ylabel('RMS shift / clean RMS radius')
 f.suptitle('Relative perturbation movement of frozen representations');f.tight_layout();fig('representation_stress_movement',f)
 f,axs=plt.subplots(2,4,figsize=(13,7));sp=pd.DataFrame(scatter)
 for i,k in enumerate(['AUC','AP']):
  for j,xname in enumerate(['MMD_common','PR','separation_ratio','probe_separation_ratio']):
   ax=axs[i,j];a=sp[(sp.kind=='paired_change')&(sp.dose==.1)&(sp.x==xname)&(sp.y==k)];ax.scatter(a.x_value,a.y_value,color='#0072B2');
   for q in a.itertuples():ax.annotate(str(q.seed),(q.x_value,q.y_value),fontsize=6)
   ax.axhline(0,c='grey',lw=.6);ax.axvline(0,c='grey',lw=.6);ax.set_xlabel('Anchored−symmetric '+xname);ax.set_ylabel('Anchored−symmetric '+k)
 f.suptitle('Paired diagnostic associations: ten seeds, no mediation identification');f.tight_layout();fig('paired_mechanism_associations',f)
 # All stresses, separate arms: no pooled arm/configuration correlation.
 f,axs=plt.subplots(8,3,figsize=(12,22))
 for i,cond in enumerate(conds):
  for j,arm in enumerate(ARMS):
   ax=axs[i,j];a=v[(v.condition==cond)&(v.arm==arm)&v.dose.isin([0,.1])]
   ax.scatter(a.MMD_common,a.rms_relative,c=COLORS[arm],s=18)
   for q in a.itertuples():ax.annotate(str(q.seed),(q.MMD_common,q.rms_relative),fontsize=6)
   ax.set_title(arm+' / '+cond,fontsize=9);ax.set_xlabel('Common-kernel cohort MMD');ax.set_ylabel('Relative representation shift')
 f.suptitle('Matching versus perturbation invariance: within-arm seed diagnostics, dose0/.1');f.tight_layout();fig('matching_vs_movement',f)
 if include_dose:
  f,axs=plt.subplots(2,4,figsize=(14,7));fs=KEYS+['PR','separation_ratio','MMD_common','trace']
  for ax,k in zip(axs.ravel(),fs):
   for arm in ['symmetric','anchored']:
    aa=pd.concat([r[r.arm=='none'],r[r.arm==arm]]);gg=aa.groupby('dose')[k];means=gg.mean();sd0=gg.std();hw=stats.t.ppf(.975,9)*sd0/np.sqrt(10)
    ax.errorbar(means.index,means,yerr=hw,color=COLORS[arm],marker='o',capsize=2,label=arm)
    for seed in SEEDS:
     t=aa[aa.seed==seed].sort_values('dose');ax.plot(t.dose,t[k],color=COLORS[arm],alpha=.1,lw=.5)
   ax.set_title(k);ax.set_xlabel('Common alignment coefficient');ax.set_xticks([0,.1,.3])
  axs[0,0].legend(fontsize=7);f.suptitle('Coarse dose challenge; shared zero reference, nominal seed intervals, no dose selection');f.tight_layout();fig('alignment_dose_challenge',f)
  # All absolute stress conditions at all doses, without selecting a favorable perturbation.
  ds=[]
  for (dose,arm,condition),g in metrics.groupby(['dose','arm','condition']):
   for k in KEYS:ds.append(dict(dose=dose,arm=arm,condition=condition,endpoint=k,**summary(g[k])))
  table('dose_absolute_stress_summary.csv',ds)
  for k in ['AUC','AP']:
   f,axs=plt.subplots(2,4,figsize=(14,7))
   for ax,condition in zip(axs.ravel(),conds):
    for arm in ['symmetric','anchored']:
     a=metrics[(metrics.condition==condition)&metrics.arm.isin(['none',arm])].groupby('dose')[k].mean();ax.plot(a.index,a.values,c=COLORS[arm],marker='o',label=arm)
    ax.set_title(condition);ax.set_xlabel('Alignment coefficient');ax.set_ylabel('Absolute stressed '+k)
   axs[0,0].legend();f.tight_layout();fig('dose_stress_'+k,f)
 print('Mechanism summaries and figures complete, include_dose=',include_dose,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--include-dose',action='store_true');a=ap.parse_args();summarize(a.include_dose)
