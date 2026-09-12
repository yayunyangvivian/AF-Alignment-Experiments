# preprocessing_PTBXL.py
# PTB-XL preprocessing (Leakage-free):
# - ECG handcrafted features -> StandardScaler (fit train only) -> PCA (fit train only)
# - Tabular: age/sex (+optional device)
# - Optional TF-IDF "text": by default only from tabular tokens (NO scp_codes to avoid label leakage)
# - Splits: prefer official strat_fold (train:1-8, val:9, test:10)

import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import wfdb
from scipy import stats
from scipy.signal import welch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ----------------------------- path helpers -----------------------------

def resolve_ptb_root(user_path: str) -> str:
    from pathlib import Path
    root=Path(user_path).expanduser()
    if not user_path or not (root/'ptbxl_database.csv').is_file():
        raise FileNotFoundError('Supply --ptb-root containing ptbxl_database.csv')
    return str(root)


# --------------------------- label/text utils ---------------------------

def parse_scp_codes(code_str):
    """PTB-XL stores scp_codes as a stringified dict; return codes sorted by weight desc."""
    try:
        d = ast.literal_eval(code_str)
        if isinstance(d, dict):
            return [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]
    except Exception:
        pass
    return []


def is_af_from_codes(code_str) -> int:
    """AF label: 1 if AFIB or AFLT appears, else 0."""
    codes = parse_scp_codes(code_str)
    return int(any(c in ("AFIB", "AFLT") for c in codes))


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def build_tabular(row: pd.Series, use_device: bool = False):
    """Return numeric tabular vector: [age, sex(0/1/2), (optional) device_id_hash]."""
    age = safe_int(row.get("age", np.nan), default=-1)
    sex = row.get("sex", np.nan)
    # PTB-XL sex often is 0/1 (female/male) or NaN. We'll map NaN->-1
    sex_i = safe_int(sex, default=-1)

    feats = [age, sex_i]

    if use_device:
        # device may be string-like; hash to stable int bucket
        dev = row.get("device", "")
        dev_s = "" if pd.isna(dev) else str(dev)
        dev_id = (hash(dev_s) % 1000) if dev_s else -1
        feats.append(dev_id)

    return np.array(feats, dtype=np.float32)


def build_doc_from_row(row: pd.Series, use_scp_in_text: bool = False, use_device: bool = False) -> str:
    """
    Tiny 'document' for TF-IDF.
    Default: ONLY demographics tokens (avoid label leakage).
    If use_scp_in_text=True, include scp_codes tokens (NOT recommended for AF task).
    """
    parts = []
    age = row.get("age", np.nan)
    if not pd.isna(age):
        parts.append(f"age_{safe_int(age)//5*5}")
    sex = row.get("sex", np.nan)
    if not pd.isna(sex):
        parts.append(f"sex_{safe_int(sex)}")

    if use_device:
        dev = row.get("device", "")
        if not pd.isna(dev) and str(dev).strip():
            parts.append(f"device_{str(dev).strip().lower()}")

    if use_scp_in_text:
        # WARNING: may leak label if task label is derived from scp_codes
        parts += [f"code_{c.lower()}" for c in parse_scp_codes(row.get("scp_codes", ""))]

    return " ".join(parts)


# -------------------------- ECG feature engineering ---------------------

def bandpower_welch(x, fs, bands=((0.5, 4), (4, 15), (15, 40))):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    total = np.trapezoid(Pxx, f) + 1e-12
    feats = []
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        feats.append(float(np.trapezoid(Pxx[m], f[m]) / total))
    return np.array(feats, dtype=np.float32)


def ecg_features_1lead(x, fs=100):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.median(x)
    s = np.median(np.abs(x)) + 1e-6
    zx = x / s

    feats = [
        float(np.mean(zx)),
        float(np.std(zx)),
        float(stats.skew(zx, bias=False)),
        float(stats.kurtosis(zx, bias=False, fisher=True)),
        float(np.quantile(zx, 0.1)),
        float(np.quantile(zx, 0.5)),
        float(np.quantile(zx, 0.9)),
        float(np.mean(np.diff(zx) ** 2)),
        float(np.mean(np.abs(np.diff(zx)))),
        float(np.mean(zx ** 2)),
        float(((zx[:-1] * zx[1:]) < 0).mean()),
    ]
    feats += bandpower_welch(zx, fs=fs).tolist()
    feats = np.array(feats, dtype=np.float32)
    feats[~np.isfinite(feats)] = 0.0
    return feats  # 14 dims


def ecg_features_12lead(sig, fs=100):
    """sig: (n_samples, 12) or (12, n_samples) -> (168,)"""
    if sig.ndim != 2:
        raise ValueError(f"Unexpected signal shape: {sig.shape}")
    if sig.shape[0] == 12 and sig.shape[1] != 12:
        sig = sig.T
    if sig.shape[1] < 12:
        pad = np.zeros((sig.shape[0], 12 - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)
    elif sig.shape[1] > 12:
        sig = sig[:, :12]

    feats = [ecg_features_1lead(sig[:, i], fs=fs) for i in range(12)]
    v = np.concatenate(feats, axis=0).astype(np.float32)
    v[~np.isfinite(v)] = 0.0
    return v  # 168


# ------------------------------- main -----------------------------------

def make_splits(df: pd.DataFrame, y_af: np.ndarray, N: int, seed: int):
    """Prefer official strat_fold: train 1-8, val 9, test 10."""
    idx_all = np.arange(N)

    if "strat_fold" in df.columns:
        folds = df["strat_fold"].values[:N]
        idx_test = idx_all[folds == 10]
        idx_val = idx_all[folds == 9]
        idx_train = idx_all[(folds != 10) & (folds != 9)]
        return idx_train, idx_val, idx_test

    # fallback: random stratified
    tr, tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_af[:N], test_size=0.2, random_state=seed, stratify=y_af[:N]
    )
    va, te, y_va, y_te = train_test_split(
        tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    return tr, va, te


def export_ptbxl_embeddings(
    ptb_root,
    out_dir,
    fs_choice=100,
    pca_dim=256,
    seed=42,
    limit=0,
    tfidf_dim=1000,
    use_scp_in_text=False,
    use_device=False,
):
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    root = resolve_ptb_root(ptb_root)
    db_csv = os.path.join(root, "ptbxl_database.csv")
    df = pd.read_csv(db_csv)

    rec_col = "filename_lr" if fs_choice == 100 else "filename_hr"
    if rec_col not in df.columns:
        raise ValueError(f"Column {rec_col} not found in ptbxl_database.csv")

    rec_paths = df[rec_col].values
    N_all = len(rec_paths)
    N = N_all if limit <= 0 else min(limit, N_all)

    # label (from scp_codes)
    y_af = df["scp_codes"].apply(is_af_from_codes).values.astype(np.int32)[:N]

    # splits first (important for leakage-free fitting)
    idx_train, idx_val, idx_test = make_splits(df, y_af, N, seed)

    # -------------------- Tabular --------------------
    tab = np.stack([build_tabular(df.iloc[i], use_device=use_device) for i in range(N)], axis=0).astype(np.float32)
    np.save(os.path.join(out_dir, "tab.npy"), tab)

    # -------------------- Optional TF-IDF (fit on train only) --------------------
    docs = df.iloc[:N].apply(lambda r: build_doc_from_row(r, use_scp_in_text=use_scp_in_text, use_device=use_device), axis=1).tolist()
    tfidf = TfidfVectorizer(max_features=tfidf_dim, ngram_range=(1, 2))

    X_text_train = tfidf.fit_transform([docs[i] for i in idx_train])
    X_text_val   = tfidf.transform([docs[i] for i in idx_val])
    X_text_test  = tfidf.transform([docs[i] for i in idx_test])

    # Save vocab (JSON-safe)
    vocab = {str(k): int(v) for k, v in tfidf.vocabulary_.items()}
    with open(os.path.join(out_dir, "tfidf_vocab.json"), "w") as f:
        json.dump(vocab, f)
    # Save idf too (useful for reproducibility)
    np.save(os.path.join(out_dir, "tfidf_idf.npy"), tfidf.idf_.astype(np.float32))

    # Reconstruct full matrix in original order (optional, convenient)
    X_text_full = np.zeros((N, X_text_train.shape[1]), dtype=np.float32)
    X_text_full[idx_train] = X_text_train.toarray().astype(np.float32)
    X_text_full[idx_val]   = X_text_val.toarray().astype(np.float32)
    X_text_full[idx_test]  = X_text_test.toarray().astype(np.float32)
    np.save(os.path.join(out_dir, "emb_text_tfidf.npy"), X_text_full)

    # -------------------- Read ECGs + handcrafted features --------------------
    feats = []
    fails = 0
    for i in tqdm(range(N), desc=f"Extract PTB-XL ECG features ({fs_choice}Hz)"):
        rel = rec_paths[i]
        full = os.path.join(root, rel)
        try:
            sig, meta = wfdb.rdsamp(full)
            fs = int(meta["fs"]) if ("fs" in meta and meta["fs"]) else fs_choice
            feats.append(ecg_features_12lead(sig, fs=fs))
        except Exception:
            fails += 1
            feats.append(np.zeros(168, dtype=np.float32))

    Z_raw = np.stack(feats, axis=0).astype(np.float32)  # (N,168)

    # Defensive cleanup
    Z_raw = np.where(np.isfinite(Z_raw), Z_raw, np.nan)
    col_means = np.nanmean(Z_raw, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(~np.isfinite(Z_raw))
    Z_raw[inds] = np.take(col_means, inds[1])

    # -------------------- Leakage-free Standardize + PCA --------------------
    scaler = StandardScaler().fit(Z_raw[idx_train])
    Z_std = scaler.transform(Z_raw).astype(np.float32)

    max_comp = int(min(pca_dim, Z_std.shape[1], max(1, len(idx_train) - 1)))
    pca = PCA(n_components=max_comp, random_state=seed, svd_solver="full")
    pca.fit(Z_std[idx_train])
    Z_ecg = pca.transform(Z_std).astype(np.float32)

    # save arrays
    np.save(os.path.join(out_dir, "emb_ecg.npy"), Z_ecg)
    np.save(os.path.join(out_dir, "y_af.npy"), y_af.astype(np.int32))

    np.savetxt(os.path.join(out_dir, "ids_train.txt"), idx_train, fmt="%d")
    np.savetxt(os.path.join(out_dir, "ids_val.txt"), idx_val, fmt="%d")
    np.savetxt(os.path.join(out_dir, "ids_test.txt"), idx_test, fmt="%d")

    np.savez(
        os.path.join(out_dir, "norm_ecg.npz"),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
    )

    print("PTB-XL DONE (leakage-free):",
          f"\n  ECG embedding: {Z_ecg.shape}",
          f"\n  TAB:           {tab.shape}",
          f"\n  TEXT TF-IDF:   {X_text_full.shape} (use_scp_in_text={use_scp_in_text})",
          f"\n  y_af:          {y_af.shape} (pos={int(y_af.sum())})",
          f"\n  splits         train:{len(idx_train)}  val:{len(idx_val)}  test:{len(idx_test)}",
          f"\n  failed reads:  {fails}",
          f"\n  PCA comps:     {max_comp}",
          f"\n  out_dir:       {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="PTB-XL preprocessing (leakage-free)")
    ap.add_argument("--ptb-root", type=str, required=True,
                    help="Folder that contains ptbxl_database.csv (e.g., .../ptb-xl/1.0.3).")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--fs", type=int, default=100, choices=[100, 500], help="Sampling split: 100 or 500 Hz.")
    ap.add_argument("--pca-dim", type=int, default=256, help="Target ECG embedding dim (auto-clamped by train size).")
    ap.add_argument("--tfidf-dim", type=int, default=1000, help="TF-IDF max features.")
    ap.add_argument("--use-scp-in-text", action="store_true",
                    help="Include scp_codes tokens in TF-IDF doc (NOT recommended; may leak label for AF).")
    ap.add_argument("--use-device", action="store_true", help="Include device token/features if available.")
    ap.add_argument("--limit", type=int, default=0, help="Process first N records if > 0 (quick sanity run).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    export_ptbxl_embeddings(
        ptb_root=args.ptb_root,
        out_dir=args.out_dir,
        fs_choice=args.fs,
        pca_dim=args.pca_dim,
        seed=args.seed,
        limit=args.limit,
        tfidf_dim=args.tfidf_dim,
        use_scp_in_text=args.use_scp_in_text,
        use_device=args.use_device,
    )


if __name__ == "__main__":
    main()
