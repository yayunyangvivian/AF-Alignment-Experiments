from pathlib import Path
import warnings
import numpy as np,pandas as pd,wfdb
from scipy.signal import resample_poly
from . import preprocess_ptb as T
LEADS=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
DATA=None
N=None
def transform(raw):
    z=(raw-N['mean'])/N['scale']
    return (z@N['pca_components'].T-N['pca_mean']@N['pca_components'].T).astype(np.float32)

def one(rel):
    p=DATA/rel
    try:
        h=wfdb.rdheader(str(p.with_suffix('')))
        assert h.n_sig==12 and h.fs==500 and 2500<=h.sig_len<=5000, 'Unexpected signal geometry'
        order=[s.upper() for s in h.sig_name];assert len(set(order))==12 and set(order)==set(LEADS),'Missing/repeated lead'
        assert all(u=='mV' for u in h.units),'Unexpected physical unit'
        sig,_=wfdb.rdsamp(str(p.with_suffix('')));assert np.isfinite(sig).all(),'Nonfinite waveform'
        sig=resample_poly(sig[:,[order.index(s) for s in LEADS]],1,5,axis=0)
        comments={c.split(':',1)[0].strip():c.split(':',1)[1].strip() for c in h.comments if ':' in c}
        dx=[s.strip() for s in comments.get('Dx','').split(',') if s.strip()]
        assert dx and all(d.isdigit() for d in dx),'Missing/invalid diagnoses'
        try:age=int(float(comments.get('Age','nan')))
        except (ValueError,OverflowError):age=-1
        sex={'male':0,'female':1}.get(comments.get('Sex','').lower(),-1)
        with warnings.catch_warnings(record=True) as ww:
            f=T.ecg_features_12lead(sig,100)
        assert np.isfinite(f).all()
        row=dict(record_id=p.stem,relative_path=rel,label=int(bool(set(dx)&{'164889003','164890007'})),age=age,sex=sex,samples_original=h.sig_len,sampling_rate_original=h.fs,seconds=h.sig_len/h.fs,diagnosis_codes=';'.join(dx),constant_leads=int(np.sum(np.std(sig,axis=0)==0)),feature_warnings=len(ww),patient_id_status='not supplied; record ID is not assumed to be a patient ID')
        return row,f,[age,sex],None
    except Exception as e:return None,None,None,dict(path=rel,error=repr(e))
