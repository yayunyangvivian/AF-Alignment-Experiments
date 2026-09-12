from __future__ import annotations

import argparse

import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, confusion_matrix,
)

def load_ids(path: str) -> np.ndarray:
    a = np.loadtxt(path, dtype=int)
    return a.ravel() if a.ndim > 1 else a

def normalize_rows(H: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)

def build_x_rich(ptbxl_csv: str) -> np.ndarray:
    df = pd.read_csv(ptbxl_csv)
    h_m = df["height"] / 100.0
    bmi = df["weight"] / (h_m ** 2)
    feats = pd.DataFrame({
        "age":             df["age"].fillna(df["age"].median()),
        "sex":             df["sex"].fillna(0),
        "height":          df["height"].fillna(df["height"].median()),
        "height_missing":  df["height"].isna().astype(float),
        "weight":          df["weight"].fillna(df["weight"].median()),
        "weight_missing":  df["weight"].isna().astype(float),
        "bmi":             bmi.fillna(bmi.median()),
        "bmi_missing":     bmi.isna().astype(float),
    }).astype(np.float32)
    return feats.values

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_val, y_val,
              hidden=128, dropout=0.2, lr=1e-3, wd=1e-4,
              epochs=80, bs=256, patience=10, device="cpu"):
    """
    Identical training loop to eval_clean_ablation.py: no internal reseeding.
    The caller seeds torch+numpy ONCE before the first call; the global RNG
    state advances through all subsequent rows in a deterministic order.
    """
    sc = StandardScaler().fit(X_tr)
    X_tr_t  = torch.from_numpy(sc.transform(X_tr)).float().to(device)
    X_val_t = torch.from_numpy(sc.transform(X_val)).float().to(device)
    y_tr_t  = torch.from_numpy(y_tr).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    model = MLP(X_tr.shape[1], hidden, dropout).to(device)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)

    best_val = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(X_val_t), y_val_t).item()
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    model.load_state_dict(best_state); model.eval()
    def predict(X_np):
        with torch.no_grad():
            xt = torch.from_numpy(sc.transform(X_np)).float().to(device)
            return torch.sigmoid(model(xt)).cpu().numpy()
    return predict

def best_f1_threshold_on_val(y_val, p_val, grid=400):
    ts = np.linspace(0.001, 0.999, grid)
    best_t, best_f = 0.5, -1.0
    for t in ts:
        f = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)

def threshold_for_sens_on_val(y_val, p_val, target=0.80):
    pos = y_val == 1
    if pos.sum() == 0:
        return float("nan")
    pos_scores = np.sort(p_val[pos])
    n_pos = int(pos.sum())
    k = int(np.ceil(target * n_pos))
    if k <= 0:    return float(pos_scores[-1])
    if k > n_pos: return float("nan")
    return float(pos_scores[n_pos - k])

def operating_stats(y_true, y_prob, thr):
    y = y_true.astype(int)
    yhat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    return dict(Sens=round(sens, 4), Spec=round(spec, 4),
                PPV=round(ppv, 4), NPV=round(npv, 4))

def bootstrap_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_delta_ci(y, p_a, p_b, fn, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        try:
            diffs.append(fn(y[idx], p_b[idx]) - fn(y[idx], p_a[idx]))
        except Exception:
            pass
    diffs = np.asarray(diffs)
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))
