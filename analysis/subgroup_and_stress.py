from common import *
import itertools

def subgroups():
 d=pd.read_csv(B/'subgroup_per_seed_metrics.csv');counts=pd.read_csv(B/'subgroup_counts.csv');table('subgroup_counts.csv',counts)
 rows=[];summ=[]
 groups=[('conduction','conduction_absent','conduction_present'),('sex','female','male'),('age','age18_64','age65plus')]
 for label,a,b in groups:
  for k in KEYS:
   for seed in SEEDS:
    def value(group,arm):return d[(d.subgroup==group)&(d.arm==arm)&(d.seed==seed)][k].item()
    da=value(a,'anchored')-value(a,'none');db=value(b,'anchored')-value(b,'none');h=da-db
    rows.append(dict(group_pair=label,group_A=a,group_B=b,seed=seed,endpoint=k,delta_A=da,delta_B=db,H=h,H_favorable=sign(k)*h))
   x=pd.DataFrame(rows);x=x[(x.group_pair==label)&(x.endpoint==k)]
   for batch,ix in [('all10',x.seed>=77),('original5',x.seed<82),('added5',x.seed>=82)]:
    g=x.loc[ix,'H'].to_numpy();summ.append(dict(group_pair=label,group_A=a,group_B=b,endpoint=k,batch=batch,**summary(g),sign_symmetry_p=flip(g),test_scope='Exact enumeration under independent sign-symmetric null only; not design-based randomization'))
 x=table('subgroup_seed_effects.csv',rows);s=table('subgroup_interaction_summary.csv',summ)
 f,axs=plt.subplots(3,4,figsize=(13,8))
 for i,(name,a,b) in enumerate(groups):
  for j,k in enumerate(KEYS):
   ax=axs[i,j];v=x[(x.group_pair==name)&(x.endpoint==k)];g=v.H_favorable.to_numpy();z=summary(g)
   ax.scatter(v.seed,g,c=['#0072B2' if t<82 else '#D55E00' for t in v.seed],s=20);ax.errorbar(89,z['mean'],yerr=[[z['mean']-z['lower']],[z['upper']-z['mean']]],fmt='ks',capsize=3)
   ax.axhline(0,c='grey',lw=.8);ax.set_title(f'{name}: {k}');ax.set_xticks([77,81,85,89],['77','81','85','mean']);ax.set_ylabel('H, favorable-direction scale' if j==0 else '')
 f.suptitle('Anchored−none: between-subgroup contrast, A minus B\nA: conduction absent / female / age18–64. Blue: original5; orange: added5; bars: nominal t9.',y=1.02);f.tight_layout();fig('subgroup_interaction',f)

def stresses():
 d=pd.read_csv(B/'stress_per_seed_metrics.csv');base=d[d.condition=='clean'].set_index(['seed','arm']);rows=[]
 for r in d.itertuples():
  for k in KEYS:
   clean=base.loc[(r.seed,r.arm),k];st=getattr(r,k);rows.append(dict(seed=r.seed,arm=r.arm,condition=r.condition,endpoint=k,clean=clean,stressed=st,change=st-clean,loss_degradation=-sign(k)*(st-clean)))
 x=table('stress_clean_absolute_degradation.csv',rows);out=[];gains=[];associations=[]
 for (condition,k),v in x[x.condition!='clean'].groupby(['condition','endpoint']):
  for cand,ref in PAIRS:
   a=v[v.arm==cand].set_index('seed');b=v[v.arm==ref].set_index('seed');assert a.index.tolist()==b.index.tolist()
   for quantity in ['clean','stressed','change','loss_degradation']:
    g=a[quantity]-b[quantity];out.append(dict(condition=condition,endpoint=k,contrast=cand+'-minus-'+ref,quantity=quantity,**summary(g)))
    for seed,h in g.items():gains.append(dict(condition=condition,endpoint=k,contrast=cand+'-minus-'+ref,quantity=quantity,seed=seed,difference=h))
  for arm in ARMS:
   vv=v[v.arm==arm];associations.append(dict(condition=condition,endpoint=k,arm=arm,K=len(vv),**corr(vv.clean,vv.loss_degradation)))
 table('stress_contrast_summary.csv',out);table('stress_seed_contrasts.csv',gains);table('clean_degradation_correlations.csv',associations)
 conditions=list(d.condition.unique());conditions.remove('clean')
 for k in ['AUC','AP']:
  f,axs=plt.subplots(2,4,figsize=(14,7),sharey=True)
  for ax,cond in zip(axs.ravel(),conditions):
   for j,arm in enumerate(ARMS):
    v=x[(x.condition==cond)&(x.endpoint==k)&(x.arm==arm)]
    for r in v.itertuples():ax.plot([j-.15,j+.15],[r.clean,r.stressed],color=COLORS[arm],alpha=.45,marker='.',lw=.7)
    ax.scatter(j-.15,v.clean.mean(),color=COLORS[arm],marker='s',s=40);ax.scatter(j+.15,v.stressed.mean(),color=COLORS[arm],marker='D',s=35)
   ax.set_title(cond);ax.set_xticks(range(3),['none','symmetric','anchored'],rotation=20);ax.set_ylabel(k)
  f.suptitle('Absolute '+k+': each line is one seed, clean (left) → stress (right)');f.tight_layout();fig('absolute_stress_'+k,f)
  f,axs=plt.subplots(2,4,figsize=(14,7))
  for ax,cond in zip(axs.ravel(),conditions):
   for arm in ARMS:
    v=x[(x.condition==cond)&(x.endpoint==k)&(x.arm==arm)];ax.scatter(v.clean,v.loss_degradation,color=COLORS[arm],label=arm,s=18)
   ax.axhline(0,color='grey',lw=.7);ax.set_title(cond);ax.set_xlabel('Clean '+k);ax.set_ylabel('Loss-coded degradation')
  axs[0,0].legend(fontsize=7);f.suptitle('Baseline–degradation coupling: descriptive only, not independent evidence of a floor');f.tight_layout();fig('clean_vs_degradation_'+k,f)
 print('Analyses1–2 complete:12 interaction rows, all stress endpoints and seeds.',flush=True)
if __name__=='__main__':check_lock();subgroups();stresses()
