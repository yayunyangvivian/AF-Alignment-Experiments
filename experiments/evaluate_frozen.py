"""Score supplied frozen routing models. Never fit on evaluation outcomes."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json
import numpy as np,pandas as pd,torch
from af_alignment import v2b
from af_alignment.scoring import predict,metrics
from af_alignment.runtime import output_dir,sha

def load_model(directory,seed,arm):
    # Only load checkpoints you trust: these use Python pickle internally.
    directory=Path(directory)
    ck=torch.load(directory/'selected_model.pt',map_location='cpu',weights_only=False)
    assert int(ck['seed'])==seed and ck['arm']==arm
    net=v2b.MaskedMultimodalTransformer(dim_ecg=168,dim_meta=2,dim_echo=512,d=128,proj_hidden=256,nhead=4,num_layers=2,ff_dim=256,dropout=.1,head_hidden=128)
    net.load_state_dict(ck['state_dict']);net.eval()
    return net,np.load(directory/'probe_parameters.npz',allow_pickle=False),ck

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',required=True,help='external_inputs.npz or stress_features.npz')
    ap.add_argument('--models',required=True,help='JSON array: seed, arm, directory, optional dose; directories relative to this JSON')
    ap.add_argument('--output',required=True)
    a=ap.parse_args();out=output_dir(a.output);torch.set_num_threads(2)
    z=np.load(a.features,allow_pickle=False);y=z['label'];x=z['ecg'];m=z['metadata']
    conditions=list(z['conditions']) if 'conditions' in z else ['clean']
    if x.ndim==2:x=x[:,None,:]
    assert x.shape[1]==len(conditions) and x.shape[0]==len(y)==len(m)
    assert np.isfinite(x).all() and set(np.unique(y))=={0,1}
    records=json.loads(Path(a.models).read_text());ids=[(r['seed'],r['arm'],r.get('dose')) for r in records]
    assert len(ids)==len(set(ids)),'Duplicate seed/arm/dose'
    rows=[];raw=[];cal=[];manifest=[]
    for r in records:
        d=Path(a.models).resolve().parent/r['directory'];net,p,ck=load_model(d,int(r['seed']),r['arm'])
        for j,condition in enumerate(conditions):
            q,c=predict(net,p,x[:,j],m)
            rows.append(dict(seed=r['seed'],arm=r['arm'],dose=r.get('dose',0 if r['arm']=='none' else .1),condition=str(condition),selected_epoch=ck['selected_epoch'],**metrics(y,q,c)))
            raw.append(q);cal.append(c)
        manifest.append(dict(seed=r['seed'],arm=r['arm'],checkpoint_sha256=sha(d/'selected_model.pt'),probe_sha256=sha(d/'probe_parameters.npz')))
    pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    np.savez_compressed(out/'predictions.npz',raw_probability=raw,calibrated_probability=cal,seed=[r['seed'] for r in rows],arm=[r['arm'] for r in rows],condition=[r['condition'] for r in rows],dose=[r['dose'] for r in rows],label=y,record_id=z['record_id'])
    (out/'execution.json').write_text(json.dumps(dict(features_sha256=sha(a.features),models=manifest,records=len(y),fitting=False),indent=2)+'\n')
if __name__=='__main__':main()
