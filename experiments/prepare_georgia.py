"""Apply the retained PTB transform; never refit it on Georgia."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json
import numpy as np,pandas as pd
from af_alignment.runtime import output_dir,sha
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,help='Georgia root with downloaded WFDB records')
    ap.add_argument('--normalizer',required=True,help='retained PTB norm_ecg.npz')
    ap.add_argument('--output',required=True)
    ap.add_argument('--stresses',action='store_true')
    a=ap.parse_args()
    from af_alignment import georgia_features as F,stress
    out=output_dir(a.output);F.DATA=Path(a.data);F.N=np.load(a.normalizer,allow_pickle=False)
    paths=sorted(str(p.relative_to(F.DATA)) for p in F.DATA.rglob('*.hea'))
    assert len(paths)==10344,'Expected complete locked Georgia release, not another cohort'
    rows=[];raw=[];meta=[];excluded=[]
    for rel in paths:
        row,f,m,error=F.one(rel)
        if error:excluded.append(error)
        else:rows.append(row);raw.append(f);meta.append(m)
    assert len(rows)==10338 and len(excluded)==6,'Unexpected exclusions; inspect rather than silently changing sample'
    d=pd.DataFrame(rows);d.to_csv(out/'record_metadata.csv',index=False);raw=np.stack(raw);x=F.transform(raw)
    assert d.label.sum()==741 and np.isfinite(x).all()
    common=dict(metadata=np.asarray(meta,dtype=np.float32),label=d.label.to_numpy(dtype=np.int32),record_id=d.record_id.to_numpy(dtype=str))
    np.savez_compressed(out/'external_inputs.npz',ecg=x,raw_features=raw,**common)
    if a.stresses:
        stressed=np.stack([stress.one(row) for row in d.itertuples(index=False)])
        sx=F.transform(stressed.reshape(-1,168)).reshape(stressed.shape)
        assert np.max(abs(stressed[:,0]-raw))==0 and np.max(abs(sx[:,0]-x))<1e-5
        np.savez_compressed(out/'stress_features.npz',ecg=sx,conditions=stress.CONDITIONS,**common)
    (out/'execution.json').write_text(json.dumps(dict(normalizer_sha256=sha(a.normalizer),listed=len(paths),included=len(rows),excluded=excluded,preprocessing_refit=False),indent=2)+'\n')
if __name__=='__main__':main()
