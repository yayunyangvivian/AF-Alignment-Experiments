"""Verify retained arithmetic, imports and gradient routing without training."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import ast,json,hashlib,subprocess
import numpy as np,pandas as pd,torch
from scipy import stats
from af_alignment import v2b as T
from af_alignment.probe import fit_probe,predict_probe
from af_alignment.scoring import predict
from af_alignment.geometry import spectrum,separation,movement

def main():
    report={};torch.set_num_threads(2)
    files=list(ROOT.rglob('*.py'))
    for p in files:compile(p.read_text(),str(p.relative_to(ROOT)),'exec')
    report['python_sources_compiled']=len(files)
    # Meaningful original gradient-routing check; no parameter optimization.
    torch.manual_seed(1007)
    h=torch.randn(1024,8,requires_grad=True);e=torch.randn(256,8,requires_grad=True);before=torch.get_rng_state().clone()
    sym=T.conditional_align(h,e,1);after=torch.get_rng_state().clone();gs=torch.autograd.grad(sum(sym),(h,e))
    torch.set_rng_state(before);anc=T.af_anchored_alignment(h,e);ga=torch.autograd.grad(sum(anc),(h,e),allow_unused=True)
    assert all(torch.equal(a,b) for a,b in zip(sym,anc)) and ga[0] is None and gs[0].abs().max()>0
    assert torch.equal(gs[1],ga[1]) and torch.equal(after,torch.get_rng_state())
    torch.set_rng_state(before);h2=h.detach().clone().requires_grad_(True);e2=e.detach().clone().requires_grad_(True)
    gh=torch.autograd.grad(h2.square().mean()+sum(T.af_anchored_alignment(h2,e2)),h2)[0]
    assert torch.allclose(gh,2*h2/h2.numel());report['gradient_routing']='passed'
    # Frozen model forward/scaler/score path with synthetic arrays only.
    net=T.MaskedMultimodalTransformer(dim_ecg=168,dim_meta=2,dim_echo=512,d=128,proj_hidden=256,nhead=4,num_layers=2,ff_dim=256,dropout=.1,head_hidden=128).eval()
    rng=np.random.default_rng(20260912);x=rng.normal(size=(8,168)).astype('float32');m=rng.normal(size=(8,2)).astype('float32')
    pp=dict(scaler_mean=np.zeros(128),scaler_scale=np.ones(128),probe_coef=np.ones((1,128))*.01,probe_intercept=np.zeros(1),calibrator_coef=np.ones((1,1)),calibrator_intercept=np.zeros(1))
    q,c=predict(net,pp,x,m);assert q.shape==c.shape==(8,) and np.isfinite(q).all() and ((c>0)&(c<1)).all()
    hh=rng.normal(size=(20,8));sp,ev=spectrum(hh.copy());assert 1<=sp['PR']<=8 and movement(hh,hh)['rms_relative']==0
    assert np.isfinite(separation(hh,np.array([0,1]*10))['separation_ratio']);report['synthetic_forward_and_geometry']='passed'
    # Recompute all retained routing and literal-three-arm mean/SD/t intervals.
    n=0
    for group in ['three_arm','three_arm_initial','routing_initial','routing_added','routing_dose03']:
        d=ROOT/'results'/group
        path=d/('per_seed_arm_metrics.csv' if group.startswith('three_arm') else 'per_seed_metrics.csv')
        scores=pd.read_csv(path)
        js=d/('paired_summaries.json' if group.startswith('three_arm') else 'summary.json')
        summaries=json.loads(js.read_text())
        if isinstance(summaries,dict):summaries=summaries.get('summaries',summaries.get('results',summaries))
        for r in summaries:
            if not isinstance(r,dict):raise ValueError('Unrecognized retained summary schema')
            gains=np.asarray(r['gains'],float);N=len(gains);sd=gains.std(ddof=1);half=stats.t.ppf(.975,N-1)*sd/np.sqrt(N)
            assert np.isclose(gains.mean(),r['mean'],rtol=0,atol=1e-12)
            assert np.isclose(sd,r['sd'],rtol=0,atol=1e-12)
            ci=r.get('nominal_t_interval',r.get('t_ci',r.get('ci')))
            assert ci is not None and np.allclose(ci,[gains.mean()-half,gains.mean()+half],rtol=0,atol=1e-12)
            endpoint=r.get('endpoint',r.get('metric'))
            possible=[]
            for a in scores.arm.unique():
                for b in scores.arm.unique():
                    if a==b:continue
                    aa=scores[scores.arm==a].set_index('seed');bb=scores[scores.arm==b].set_index('seed')
                    ids=sorted(set(aa.index)&set(bb.index))
                    if len(ids)==N:possible.append((aa.loc[ids,endpoint]-bb.loc[ids,endpoint]).to_numpy())
            assert any(np.allclose(gains,g,rtol=0,atol=1e-12) for g in possible),(group,r['contrast'])
            n+=1
    report['retained_endpoint_summaries_verified']=n
    p=ROOT/'outputs/verification';p.mkdir(parents=True,exist_ok=True)
    (p/'tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
