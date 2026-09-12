"""Re-extract frozen Kinetics features; does not recover historical execution."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json
import numpy as np
from af_alignment.runtime import output_dir,sha
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    import torch,pandas as pd
    from af_alignment import echo_features as F
    assert sha(a.weights)=='b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d'
    out=output_dir(a.output);torch.set_num_threads(1)
    d=F.build_index(a.data);assert len(d)==len(pd.read_csv(Path(a.data)/'FileList.csv'))
    net=F.r3d_18(weights=None);net.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));net.fc=torch.nn.Identity();net.eval();pre=F.R3D_18_Weights.KINETICS400_V1.transforms()
    z=np.lib.format.open_memmap(out/'emb_echo.npy',mode='w+',dtype='float32',shape=(len(d),512));rows=[]
    for i,row in enumerate(d.itertuples(index=False)):
        with torch.no_grad():v=net(F.load_and_preprocess_video(row.path,pre,32)).numpy()[0].astype('float32')
        assert v.shape==(512,) and np.isfinite(v).all();z[i]=v
        rows.append(dict(row_index=i,video_filename=Path(row.path).name,split=row.split,video_sha256=sha(row.path)))
    z.flush();np.save(out/'y_ef.npy',d.ef.to_numpy(dtype='float32'))
    for split in ['train','val','test']:np.savetxt(out/f'ids_echo_{split}.txt',np.flatnonzero(d.split.to_numpy()==split),fmt='%d')
    pd.DataFrame(rows).to_csv(out/'row_manifest.csv',index=False)
    (out/'execution.json').write_text(json.dumps(dict(rows=len(d),weights_sha256=sha(a.weights),feature_sha256=sha(out/'emb_echo.npy'),frames=32,dimension=512,new_execution=True),indent=2)+'\n')
if __name__=='__main__':main()
