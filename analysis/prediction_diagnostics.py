from common import *
from af_alignment import scoring as E
from scipy.special import expit,logit
from scipy.optimize import minimize,minimize_scalar
import argparse
from paired_helpers import paired
def subgroups():
 z=np.load(R/'clean_predictions.npz');df=pd.read_csv(X/'record_metadata.csv');assert np.array_equal(z['record_id'],df.record_id.to_numpy(dtype=str));y=z['label'];codes={'6374002','733534002','164909002','713427006','59118001','713426002','698252002','270492004','445118002'};conduction=df.diagnosis_codes.astype(str).map(lambda x:bool(set(x.split(';'))&codes)).to_numpy()
 masks={'male':(df.sex==0).to_numpy(),'female':(df.sex==1).to_numpy(),'age_under18':((df.age>=0)&(df.age<18)).to_numpy(),'age18_64':((df.age>=18)&(df.age<65)).to_numpy(),'age65plus':(df.age>=65).to_numpy(),'age_unknown':(df.age<0).to_numpy(),'conduction_present':conduction,'conduction_absent':~conduction};counts=[];rows=[]
 for name,ix in masks.items():
  n=int(ix.sum());pos=int(y[ix].sum());neg=n-pos;ok=n>=200 and min(pos,neg)>=30
  counts.append(dict(subgroup=name,n=n,positives=pos,negatives=neg,prevalence=pos/n if n else None,evaluated=ok,reason='eligible' if ok else 'predefined n/class-count display threshold not met'))
  if not ok:continue
  for j,(seed,arm) in enumerate(zip(z['seed'],z['arm'])):
   if arm not in ARMS:continue
   rows.append(dict(subgroup=name,seed=int(seed),arm=str(arm),**E.metrics(y[ix],z['raw_probability'][j,ix],z['calibrated_probability'][j,ix])))
 pd.DataFrame(counts).to_csv(R/'subgroup_counts.csv',index=False);d=pd.DataFrame(rows);d.to_csv(R/'subgroup_per_seed_metrics.csv',index=False);s,g=paired(d,['subgroup'],'combined10');s.to_csv(R/'subgroup_contrasts.csv',index=False);g.to_csv(R/'subgroup_individual_gains.csv',index=False)
 h=[]
 for a,b in [('male','female'),('age65plus','age18_64'),('conduction_present','conduction_absent')]:
  for (contrast,k),v in g.groupby(['contrast','endpoint']):
   ga=v[v.subgroup==a].set_index('seed');gb=v[v.subgroup==b].set_index('seed')
   if len(ga)!=10 or len(gb)!=10:continue
   h.append(dict(group_difference=a+'-minus-'+b,contrast=contrast,endpoint=k,**summary(ga.difference-gb.difference)))
 pd.DataFrame(h).rename(columns={'candidate_favored':'positive_between_group_differences','reference_favored':'negative_between_group_differences'}).to_csv(R/'subgroup_heterogeneity.csv',index=False)

def calibration():
 z=np.load(R/'clean_predictions.npz');y=z['label'];rows=[];bins=[]
 for j,(seed,arm) in enumerate(zip(z['seed'],z['arm'])):
  if arm not in ARMS:continue
  for kind,key in [('raw','raw_probability'),('PTB_calibrated','calibrated_probability')]:
   p=z[key][j];pc=np.clip(p,1e-6,1-1e-6);xx=logit(pc);design=np.column_stack([np.ones(len(p)),xx]);
   def fg(b):
    eta=design@b;return np.mean(np.logaddexp(0,eta)-y*eta),design.T@(expit(eta)-y)/len(y)
   fit=minimize(fg,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':2000});grad=float(np.max(np.abs(fg(fit.x)[1])));assert grad<1e-6,(seed,arm,kind,fit.message,grad)
   cil=minimize_scalar(lambda alpha:np.mean(np.logaddexp(0,alpha+xx)-y*(alpha+xx)),bounds=(-20,20),method='bounded');assert cil.success
   ids=np.minimum((p*10).astype(int),9);ece=0.
   for b in range(10):
    ix=ids==b;n=int(ix.sum());ob=float(y[ix].mean()) if n else np.nan;pr=float(p[ix].mean()) if n else np.nan
    if n:ece+=n/len(p)*abs(ob-pr)
    bins.append(dict(seed=int(seed),arm=str(arm),kind=kind,bin=b,lower=b/10,upper=(b+1)/10,n=n,observed=ob,predicted=pr))
   rows.append(dict(seed=int(seed),arm=str(arm),kind=kind,intercept=float(fit.x[0]),slope=float(fit.x[1]),calibration_in_large=float(cil.x),ECE10=ece,Brier=float(np.mean((p-y)**2)),clipped=int(np.sum(pc!=p)),gradient_max=grad,diagnostic_only=True))
 pd.DataFrame(rows).to_csv(R/'calibration_per_seed.csv',index=False);pd.DataFrame(bins).to_csv(R/'reliability_bins.csv',index=False)
 pd.DataFrame(rows).groupby(['arm','kind'])[['intercept','slope','calibration_in_large','ECE10','Brier']].agg(['mean','median','min','max']).to_csv(R/'calibration_summary.csv')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,help='directory with clean_predictions.npz and record_metadata.csv');args=ap.parse_args()
 X=Path(args.input)
 if not (R/'clean_predictions.npz').exists():shutil.copyfile(X/'clean_predictions.npz',R/'clean_predictions.npz')
 subgroups();calibration()
