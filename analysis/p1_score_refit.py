import sys,os,json,csv
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import p1_support as U
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,f1_score
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();D=Path(a.output);D.mkdir(parents=True,exist_ok=False);z=np.load(a.predictions);y=z['test_label'];yv=z['validation_label'];rows=[];p=z['raw_test'];v=z['raw_validation']
for i,name in enumerate(z['method']):
 row={'method':name,'status':'new recomputation'}
 for e,fn in [('AUC',roc_auc_score),('AP',average_precision_score),('Brier',brier_score_loss)]:
  row[e]=fn(y,p[i]);row[e+'_lo'],row[e+'_hi']=U.bootstrap_ci(y,p[i],fn,n=1000,seed=42)
 row['F1']=f1_score(y,p[i]>=U.best_f1_threshold_on_val(yv,v[i]));rows.append(row)
with (D/'recomputed_benchmark.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
pairs=[]
for a,b in [(5,3),(5,6),(3,6),(1,3),(1,7),(3,4)]:
 for e,fn in [('AUC',roc_auc_score),('AP',average_precision_score)]:
  _,lo,hi=U.paired_delta_ci(y,p[a],p[b],fn,n=1000,seed=42);pairs.append(dict(base=str(z['method'][a]),comparison=str(z['method'][b]),endpoint=e,observed_difference=fn(y,p[b])-fn(y,p[a]),lo=lo,hi=hi))
(D/'recomputed_paired_contrasts.json').write_text(json.dumps(pairs,indent=2))
print(json.dumps(rows[3],indent=2))
