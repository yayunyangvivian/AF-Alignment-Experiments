import os

import numpy as np

import pandas as pd

from tqdm import tqdm

import torch

import torch.nn as nn

from torchvision.io import read_video

from torchvision.models.video import r3d_18, R3D_18_Weights

def build_index(echo_root: str):
    # 1) 找 filelist
    candidates = ["FileList.csv", "filelist.csv", "FileList.csv".lower()]
    csv_path = None
    for c in candidates:
        p = os.path.join(echo_root, c)
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"FileList.csv/filelist.csv not found in {echo_root}")

    df = pd.read_csv(csv_path)

    # 2) 猜列名
    def pick_col(options):
        for k in options:
            if k in df.columns:
                return k
        return None

    fname_col = pick_col(["Filename", "filename", "FileName", "file_name"])
    split_col = pick_col(["Split", "split", "split_name"])
    ef_col    = pick_col(["EF", "EjectionFraction", "ef"])

    if fname_col is None:
        raise ValueError(f"Filename col not found. cols={df.columns.tolist()}")
    if split_col is None:
        raise ValueError(f"Split col not found. cols={df.columns.tolist()}")
    if ef_col is None:
        raise ValueError(f"EF col not found. cols={df.columns.tolist()}")

    # 3) 找视频目录
    vid_dir = None
    for v in ["Videos", "videos", "Vids", "vids"]:
        p = os.path.join(echo_root, v)
        if os.path.isdir(p):
            vid_dir = p
            break
    if vid_dir is None:
        vid_dir = echo_root

    paths, efs, splits = [], [], []
    for r in df.itertuples(index=False):
        base = str(getattr(r, fname_col)).strip()
        sp   = str(getattr(r, split_col)).strip().lower()
        ef   = float(getattr(r, ef_col))

        cand = [
            os.path.join(vid_dir, base),
            os.path.join(vid_dir, base + ".avi"),
            os.path.join(vid_dir, base + ".mp4"),
        ]
        vp = next((c for c in cand if os.path.exists(c)), None)
        if vp is None:
            continue

        paths.append(vp)
        efs.append(ef)
        splits.append(sp)

    out = pd.DataFrame({"path": paths, "ef": efs, "split": splits}).reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("No valid video paths found. Index is empty.")
    return out

def load_and_preprocess_video(path, preprocess, num_frames=32):
    # read_video: output_format="TCHW" -> [T,C,H,W]
    v, _, _ = read_video(path, output_format="TCHW", pts_unit="sec")
    if v.numel() == 0 or v.shape[0] == 0:
        raise RuntimeError(f"Empty video: {path}")

    T = v.shape[0]
    if T < num_frames:
        reps = int(np.ceil(num_frames / T))
        v = v.repeat(reps, 1, 1, 1)
        T = v.shape[0]

    idx = torch.linspace(0, T - 1, steps=num_frames).long()
    v = v[idx]  # [num_frames, C, H, W]
    v = v.float() / 255.0

    # torchvision video weights 的 transforms 通常期望 [T,C,H,W] 或 [C,T,H,W]
    # 这里用 [T,C,H,W] 先试；如果你 torchvision 版本要求 [C,T,H,W]，就改一下 permute
    v = preprocess(v)  # -> 一般会做 resize/crop/normalize

    # 变成 [1,C,T,H,W]
    if v.shape[0] == num_frames:            # [T,C,H,W]
        x = v.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        # 如果 preprocess 输出的是 [C,T,H,W]
        x = v.unsqueeze(0)
    return x
