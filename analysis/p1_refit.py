"""New downstream refit, not historical model recovery; no encoder training."""
import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

import p1_support as U

PTB=Z_ROOT=OUT=None

def ece_score(y_true, y_prob, n_bins=15):
    """Identical definition to baseline_ptbxl.py's ece_score."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)


def platt_scale(p_val, y_val, p_test):
    """1-D Platt scaling: fit a logistic regression of y_val on the scalar
    val-set probability, apply to test. Equivalent to
    CalibratedClassifierCV(..., method='sigmoid', cv='prefit') applied to a
    probability vector rather than a fitted classifier."""
    lr = LogisticRegression()
    lr.fit(p_val.reshape(-1, 1), y_val)
    return lr.predict_proba(p_test.reshape(-1, 1))[:, 1]


def main():
    global PTB,Z_ROOT,OUT
    import argparse
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--ptb-dir',required=True);ap.add_argument('--representations',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    PTB=a.ptb_dir;Z_ROOT=a.representations;OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    id_tr = U.load_ids(os.path.join(PTB, "ids_train.txt"))
    id_va = U.load_ids(os.path.join(PTB, "ids_val.txt"))
    id_te = U.load_ids(os.path.join(PTB, "ids_test.txt"))
    y_af = np.load(os.path.join(PTB, "y_af.npy")).astype(np.float32)
    y_tr, y_va, y_te = y_af[id_tr], y_af[id_va], y_af[id_te]

    X_meta = np.load(os.path.join(PTB, "tab.npy")).astype(np.float32)
    X_rich = U.build_x_rich(os.path.join(PTB, "ptbxl_database.csv"))
    Xm_tr, Xm_va, Xm_te = X_meta[id_tr], X_meta[id_va], X_meta[id_te]
    Xr_tr, Xr_va, Xr_te = X_rich[id_tr], X_rich[id_va], X_rich[id_te]

    def load_latent(name):
        d = os.path.join(Z_ROOT, name)
        return (
            U.normalize_rows(np.load(os.path.join(d, "ptbxl_h_train.npy")).astype(np.float32)),
            U.normalize_rows(np.load(os.path.join(d, "ptbxl_h_val.npy")).astype(np.float32)),
            U.normalize_rows(np.load(os.path.join(d, "ptbxl_h_test.npy")).astype(np.float32)),
        )

    Hecg_tr, Hecg_va, Hecg_te = load_latent("Z_ECG")
    Hm_tr, Hm_va, Hm_te = load_latent("Z_meta")
    Hem_tr, Hem_va, Hem_te = load_latent("Z_ECG_meta")

    # Exactly run_unified_table1.py's own variant order, so the MLP RNG state
    # reaching row A and row D matches the run that produced Table tab:main.
    variants = [
        ("Tab (X_meta) ref", Xm_tr, Xm_va, Xm_te),
        ("B. MLP (Z_ECG only)", Hecg_tr, Hecg_va, Hecg_te),
        ("C. MLP (Z_meta only)", Hm_tr, Hm_va, Hm_te),
        ("D. MLP (X_meta + Z_ECG)",
         np.concatenate([Xm_tr, Hecg_tr], axis=1),
         np.concatenate([Xm_va, Hecg_va], axis=1),
         np.concatenate([Xm_te, Hecg_te], axis=1)),
        ("E. MLP (X_meta + Z_ECG+meta)",
         np.concatenate([Xm_tr, Hem_tr], axis=1),
         np.concatenate([Xm_va, Hem_va], axis=1),
         np.concatenate([Xm_te, Hem_te], axis=1)),
        ("A. MLP (X_rich)", Xr_tr, Xr_va, Xr_te),
        ("D'. MLP (X_rich + Z_ECG)",
         np.concatenate([Xr_tr, Hecg_tr], axis=1),
         np.concatenate([Xr_va, Hecg_va], axis=1),
         np.concatenate([Xr_te, Hecg_te], axis=1)),
        ("DIAG. MLP (Z_ECG+meta only)", Hem_tr, Hem_va, Hem_te),
    ]

    mlp_probs = {}
    for name, Xt, Xv, Xe in variants:
        predict = U.train_mlp(Xt, y_tr, Xv, y_va, epochs=80)
        mlp_probs[name] = dict(val=predict(Xv), test=predict(Xe))
        bri = brier_score_loss(y_te, mlp_probs[name]["test"])
        print(f"  {name:32s} raw Brier(test)={bri:.4f}")

    # Sanity check against Table tab:main's already-verified row values.
    bri_A = brier_score_loss(y_te, mlp_probs["A. MLP (X_rich)"]["test"])
    bri_D = brier_score_loss(y_te, mlp_probs["D. MLP (X_meta + Z_ECG)"]["test"])
    print(f"\nSanity check vs Table tab:main: row A Brier={bri_A:.4f} "
          f"(manuscript reports 0.214), row D Brier={bri_D:.4f} "
          f"(manuscript reports 0.093)")

    # LR rows (deterministic, order-independent).
    def fit_lr(Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=4000, class_weight="balanced")
        lr.fit(sc.transform(Xtr), y_tr)
        return lr, sc

    lr_rich, sc_rich = fit_lr(Xr_tr, Xr_te)
    p_lr_rich_va = lr_rich.predict_proba(sc_rich.transform(Xr_va))[:, 1]
    p_lr_rich_te = lr_rich.predict_proba(sc_rich.transform(Xr_te))[:, 1]

    lr_ecg, sc_ecg = fit_lr(Hecg_tr, Hecg_te)
    p_lr_ecg_va = lr_ecg.predict_proba(sc_ecg.transform(Hecg_va))[:, 1]
    p_lr_ecg_te = lr_ecg.predict_proba(sc_ecg.transform(Hecg_te))[:, 1]

    rows = {
        "LR (tabular only)": dict(val=p_lr_rich_va, test=p_lr_rich_te),
        "MLP (tabular only)": mlp_probs["A. MLP (X_rich)"],
        "LR (MCAT signal only)": dict(val=p_lr_ecg_va, test=p_lr_ecg_te),
        "MLP (tabular + MCAT)": mlp_probs["D. MLP (X_meta + Z_ECG)"],
    }

    np.savez_compressed(OUT/'p1_refit_predictions.npz',
        method=np.array(list(mlp_probs)),raw_test=np.array([v['test'] for v in mlp_probs.values()]),
        raw_validation=np.array([v['val'] for v in mlp_probs.values()]),
        calibration_method=np.array(list(rows)),
        calibration_raw_test=np.array([v['test'] for v in rows.values()]),
        calibration_test=np.array([platt_scale(v['val'], y_va, v['test']) for v in rows.values()]),
        test_label=y_te,validation_label=y_va,test_record_index=id_te,validation_record_index=id_va,
        test_patient_id=np.load(Path(PTB)/'test_patient_ids.npy'))
    out = {}
    print("\n=== Table tab:calib regeneration (unified pipeline, seed=42) ===")
    for name, pr in rows.items():
        p_va, p_te = pr["val"], pr["test"]
        bri_raw = brier_score_loss(y_te, p_te)
        ece_raw = ece_score(y_te, p_te)
        p_te_platt = platt_scale(p_va, y_va, p_te)
        bri_platt = brier_score_loss(y_te, p_te_platt)
        ece_platt = ece_score(y_te, p_te_platt)
        out[name] = dict(brier_raw=bri_raw, ece_raw=ece_raw,
                          brier_platt=bri_platt, ece_platt=ece_platt)
        print(f"{name:24s} Brier_raw={bri_raw:.4f} ECE_raw={ece_raw:.3f} "
              f"Brier_Platt={bri_platt:.4f} ECE_Platt={ece_platt:.3f}")

    with open(OUT/"calibration_scores.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved regenerate_tab_calib_output.json")


if __name__ == "__main__":
    main()
