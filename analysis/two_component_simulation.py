from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
#!/usr/bin/env python3
"""2026-09-07 replacement calculations; not recovery of historical producing runs.
Standalone: Python, NumPy, SciPy only. Run from any directory; outputs beside script.
The legacy coverage/power RNG stream is intentionally preserved, without tuning.
"""
from pathlib import Path
import hashlib, json, sys, platform
import numpy as np
import scipy
from scipy import stats, optimize
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
from af_alignment.runtime import output_dir
ROOT=output_dir(args.output)
CONFIG = dict(sigma_gain=0.01309220001610004, sigma_test=0.009124715211419838,
              M=20000, M_bootstrap=5000, B=999, Delta=0.007, seed=2024,
              bit_generator='PCG64', quantile_method='linear', alpha=0.05)
se = float(np.sqrt(CONFIG['sigma_gain']**2-CONFIG['sigma_test']**2))
st = CONFIG['sigma_test']; M=CONFIG['M']; B=CONFIG['B']; Delta=CONFIG['Delta']
rng = np.random.default_rng(CONFIG['seed'])
rows=[]; old_power=[]; changes=[]

def draw(D,K,skew,m=M):
    if skew:
        a=(rng.exponential(1,(m,K))-1)*se
        b=(rng.exponential(1,(m,K))-1)*st
    else:
        a=rng.normal(0,se,(m,K)); b=rng.normal(0,st,(m,K))
    return D+a+b

def proportion(x):
    count=int(np.count_nonzero(x)); n=int(np.size(x)); p=count/n
    return dict(count=count,n=n,p=p,percent=100*p,mcse_pp=100*np.sqrt(p*(1-p)/n),display=f'{100*p:.1f}')

def add(law,method,K,cov,rej,**kw):
    rows.append(dict(law=law,method=method,K=K,mean_coverage=proportion(cov),mean_null_rejection=proportion(rej),**kw))

# Sum of two independent exponentials with distinct scales; centered to mean zero.
def skew_cdf(x):
    y=x+se+st
    if y<=0: return 0.
    return 1-(se*np.exp(-y/se)-st*np.exp(-y/st))/(se-st)
median_skew=float(optimize.brentq(lambda x:skew_cdf(x)-.5,-se-st,10*(se+st),xtol=1e-15))
for skew,law in ((False,'Gaussian'),(True,'shifted_exponential_components')):
    h=1.96*st
    add(law,'naive',1,np.abs(draw(Delta,1,skew)[:,0]-Delta)<h,np.abs(draw(0,1,skew)[:,0])>h)
    for K in (5,10):
        g=draw(Delta,K,skew); hh=stats.t.ppf(.975,K-1)*g.std(1,ddof=1)/np.sqrt(K)
        g0=draw(0,K,skew); h0=stats.t.ppf(.975,K-1)*g0.std(1,ddof=1)/np.sqrt(K)
        add(law,'t',K,np.abs(g.mean(1)-Delta)<hh,np.abs(g0.mean(1))>h0)
    for method,stud in (('percentile',False),('studentized',True)):
        res=[]; zero_count=[]
        for D in (Delta,0.):
            g=draw(D,10,skew,CONFIG['M_bootstrap'])
            idx=rng.integers(0,10,(CONFIG['M_bootstrap'],B,10))
            bsamp=np.take_along_axis(g[:,None,:],idx,axis=2)
            if stud:
                mm,ss=g.mean(1),g.std(1,ddof=1)/np.sqrt(10)
                bsd=bsamp.std(2,ddof=1)
                T=(bsamp.mean(2)-mm[:,None])/np.where(bsd>0,bsd/np.sqrt(10),np.inf)
                zero_count.append(int((bsd==0).sum()))
                qh,ql=np.percentile(T,[97.5,2.5],axis=1)
                res.append((mm-qh*ss,mm-ql*ss))
            else:
                res.append(np.percentile(bsamp.mean(2),[2.5,97.5],axis=1))
        (lo,hi),(lo0,hi0)=res
        add(law,method,10,(lo<=Delta)&(Delta<=hi),~((lo0<=0)&(0<=hi0)),inner_bootstrap=B,zero_se_count_coverage_null=zero_count)
        del bsamp,idx
    K=10; k=1
    g=np.sort(draw(Delta,K,skew),axis=1); g0=draw(0,K,skew)
    npos=(g0>0).sum(1); med=median_skew if skew else 0.
    # Additional summaries reuse exactly the existing sign draws: no new RNG draws.
    nposmed=(g0>med).sum(1)
    add(law,'sign',K,(g[:,k]<=Delta)&(Delta<=g[:,K-1-k]),(npos<=k)|(npos>=K-k),
        median_coverage=proportion((g[:,k]<=Delta+med)&(Delta+med<=g[:,K-1-k])),
        median_null_rejection=proportion((nposmed<=k)|(nposmed>=K-k)),error_median=med)

# Audit the available generator's complete power stream, including its noisy search.
for D in (.005,.010,.020):
    rr=[]
    for K in (5,10,20):
        g=draw(D,K,False); h=stats.t.ppf(.975,K-1)*g.std(1,ddof=1)/np.sqrt(K)
        rr.append(dict(K=K,**proportion(np.abs(g.mean(1))>h)))
    search=[]
    for K in range(3,300):
        gm=draw(D,K,False,4000).mean(1)
        gs=draw(D,K,False,4000).std(1,ddof=1)
        pp=proportion(np.abs(gm)>stats.t.ppf(.975,K-1)*gs/np.sqrt(K))
        search.append(dict(K=K,**pp))
        if pp['percent']>=80: break
    old_power.append(dict(delta=D,power=rr,search=search,K80=K,normal_approx=(2.8*CONFIG['sigma_gain']/D)**2))

# Replacement main power table: exact noncentral-t probabilities, no simulation error.
def power(D,K):
    tc=stats.t.ppf(.975,K-1); nc=np.sqrt(K)*D/CONFIG['sigma_gain']
    return float(stats.nct.cdf(-tc,K-1,nc)+stats.nct.sf(tc,K-1,nc))
analytic=[]
for D in (.003,.005,.010,.020):
    km=next(K for K in range(2,10000) if power(D,K)>=.8)
    analytic.append(dict(delta=D,power=[dict(K=K,p=power(D,K),percent=100*power(D,K),display=f'{100*power(D,K):.1f}') for K in (5,10,20)],minimum_K80=km,power_previous=power(D,km-1),power_at_minimum=power(D,km),normal_approx=(2.8*CONFIG['sigma_gain']/D)**2))
old_cov=[83.2,95.,94.9,89.8,95.1,97.7,90.1,91.7,92.2,87.8,94.5,95.7]
old_rej=[17.,5.3,5.2,10.6,5.,2.2,9.7,8.5,7.4,12.4,5.9,4.3]
for i,row in enumerate(rows):
    for key,old in [('mean_coverage',old_cov[i]),('mean_null_rejection',old_rej[i])]:
        new=float(row[key]['display'])
        if new!=old: changes.append(dict(table='S6',law=row['law'],method=row['method'],K=row['K'],field=key,original_percent=old,revised_percent=new,full_precision=row[key],reason='Transparent regeneration using declared available workflow; original producing run remains unrecovered.'))
    if row['method']=='sign':
        old=97.7 if row['law']=='Gaussian' else 98.
        changes.append(dict(table='S6',law=row['law'],method='sign',field='median_coverage',original_percent=old,revised_percent=float(row['median_coverage']['display']),full_precision=row['median_coverage'],reason='New summary on same sign-interval draws, with analytic true median; replaces separately simulated historical median entry.'))
changes.extend([
 dict(table='S6',field='bootstrap_outer_simulations_per_coverage_or_null_cell',original=20000,revised=5000,reason='Actual available workflow count, now explicitly declared in regeneration; 999 inner resamples unchanged.'),
 dict(table='S6',field='MCSE_statement',original='at most 0.27 percentage points using nominal 20000 for all rows',revised='row-specific MCSE; maximum '+str(max(r[k]['mcse_pp'] for r in rows for k in ['mean_coverage','mean_null_rejection']))+' percentage points',reason='Correct method-specific denominator.'),
 dict(table='S6',field='skew_error_median',original=-.0025,revised=median_skew,reason='Analytic convolution CDF for specified full-precision scales.'),
 dict(table='S6',field='skew_median_null_rejection',original_percent=2.2,revised_percent=float(rows[-1]['median_null_rejection']['display']),full_precision=rows[-1]['median_null_rejection'],reason='Regenerated on same null draws shifted by analytic median; replaces separate historical check, not a theoretical level change.')])
old_pt=[[7,10,16],[10,19,37],[26,58,90],[72,99,100]]
old_K=['approximately 150','approximately 56','approximately 16','approximately 6']
for i,row in enumerate(analytic):
    for j,p in enumerate(row['power']):
        changes.append(dict(table='6',delta=row['delta'],K=p['K'],original_percent=old_pt[i][j],revised_percent=float(p['display']),full_precision=p['p'],reason='New analytic Gaussian noncentral-t calculation, replacing unauthenticated Monte Carlo entry; no MCSE.'))
    changes.append(dict(table='6',delta=row['delta'],field='K80',original=old_K[i],revised=row['minimum_K80'],reason='Exact minimum integer under specified Gaussian model, replacing approximate/noisy planning value.'))

def save(name,x): (ROOT/name).write_text(json.dumps(x,indent=2)+'\n')
config=dict(CONFIG,sigma_enc=se,scales_derivation='sample SD of ten Configuration-2 AUC gains; seed-42 record bootstrap SD, B=2000 RandomState(0); encoder scale is exploratory sqrt(sigma_gain^2-sigma_test^2), not an identified component',stream_order='Gaussian then skew: naive coverage/null; t K5 coverage/null; t K10 coverage/null; percentile coverage/null each draw then indices; studentized likewise; sign coverage/null; then Gaussian powers and noisy K80 searches. Analytic-median summaries consume no RNG.',generator_version='regenerate_simulations.py 2026-09-07',versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),original_producing_run_authenticated=False)
save('SIMULATION_INPUTS.json',config)
save('REGENERATED_COVERAGE.json',dict(date='2026-09-07',status='newly regenerated, not historical recovery',rows=rows,skew_error_median=median_skew,combined_skew=2*(se**3+st**3)/(se**2+st**2)**1.5))
save('AVAILABLE_GENERATOR_POWER_AUDIT.json',dict(rows=old_power,independent_gaussian_mean_sd_valid=True,reason='For Gaussian iid gains, the sample mean and sample variance are independent; independently generated normal samples yield the same required product law. The first crossing of independently noisy power estimates is not an exact minimum-K calculation.',delta_003_absent=True))
save('ANALYTIC_POWER.json',dict(method='P(T<-t_.975,K-1)+P(T>t_.975,K-1), T~noncentral-t(K-1,sqrt(K)*delta/sigma_gain)',sigma_gain=CONFIG['sigma_gain'],rows=analytic,mcse=None))
save('SIMULATION_NUMERICAL_CHANGES.json',changes)
source=ROOT/'original_reproduce_inference_tables.py'
save('SIMULATION_MANIFEST.json',dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),source,ROOT/'SIMULATION_INPUTS.json'] if p.exists()},historical_source='retained simulation generator',no_training=True,no_original_artifacts_modified=True))
print(json.dumps(dict(mean_coverage_matches=sum(float(r['mean_coverage']['display'])==o for r,o in zip(rows,old_cov)),combined_matches=sum(float(r[key]['display'])==old for key,vals in [('mean_coverage',old_cov),('mean_null_rejection',old_rej)] for r,old in zip(rows,vals)),median=median_skew,coverage_rows=rows,power=analytic),indent=2))
