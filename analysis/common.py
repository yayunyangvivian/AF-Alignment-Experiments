from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pathlib import Path
import sys, json, hashlib, importlib.util, os
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'plot_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'outputs/cache'))
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os,shutil
REPO=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('AF_ANALYSIS_OUTPUT','outputs/mechanism')).resolve()
R.mkdir(parents=True,exist_ok=True)
B=Path(os.environ.get('AF_EVALUATION_INPUT',str(REPO/'results/georgia')))
for p in (REPO/'results/mechanism').glob('*.csv'):
 if not (R/p.name).exists():shutil.copyfile(p,R/p.name)
KEYS=['AUC','AP','Brier','calibrated_Brier'];ARMS=['none','symmetric','anchored'];SEEDS=list(range(77,87))
PAIRS=[('anchored','none'),('symmetric','none'),('anchored','symmetric')]
COLORS={'none':'#666666','symmetric':'#D55E00','anchored':'#0072B2'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(name,obj):
 (R/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def table(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(R/name,index=False);return d
def fig(name,f):
 f.savefig(R/(name+'.pdf'),bbox_inches='tight');f.savefig(R/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(f)
def sign(k):return 1 if k in ['AUC','AP'] else -1
def summary(g):
 g=np.asarray(g,float);n=len(g);assert n>=2 and np.isfinite(g).all();m=g.mean();sd=g.std(ddof=1);hw=stats.t.ppf(.975,n-1)*sd/np.sqrt(n)
 return dict(K=n,mean=m,median=np.median(g),sd=sd,lower=m-hw,upper=m+hw,positive=int((g>0).sum()),negative=int((g<0).sum()),zeros=int((g==0).sum()))
def flip(g):
 g=np.asarray(g,float);ss=2*((np.arange(2**len(g))[:,None]>>np.arange(len(g)))&1)-1
 return float(np.mean(np.abs(ss@g/len(g))>=abs(g.mean())-1e-14))
def corr(x,y):
 x=np.asarray(x);y=np.asarray(y)
 if np.std(x)==0 or np.std(y)==0:return dict(pearson=np.nan,spearman=np.nan)
 return dict(pearson=stats.pearsonr(x,y).statistic,spearman=stats.spearmanr(x,y).statistic)
def check_lock():
 from af_alignment.runtime import sha
 manifest=json.loads((REPO/'provenance/file_manifest.json').read_text())
 for name,h in manifest.items():
  if name.startswith('results/'):assert sha(REPO/name)==h,name
def markdown(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
