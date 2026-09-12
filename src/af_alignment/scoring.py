import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
def predict(net,pp,x,meta):
    with torch.no_grad():
        h=np.concatenate([net(z_ecg=torch.from_numpy(x[i:i+1024]),z_meta=torch.from_numpy(meta[i:i+1024]))[0].numpy() for i in range(0,len(x),1024)])
    h=h/(np.linalg.norm(h,axis=1,keepdims=True)+1e-8)
    # Match sklearn StandardScaler's float32 in-place transform rounding.
    h-=pp['scaler_mean'].astype(h.dtype);h/=pp['scaler_scale'].astype(h.dtype)
    prob=expit((h@pp['probe_coef'].T+pp['probe_intercept']).ravel())
    cal=expit((prob[:,None]@pp['calibrator_coef'].T+pp['calibrator_intercept']).ravel())
    return prob,cal

def metrics(y,p,c):
    return dict(AUC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),calibrated_Brier=float(brier_score_loss(y,c)))
