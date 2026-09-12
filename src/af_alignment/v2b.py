import os

import argparse

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

def load_ids_txt(path: str) -> np.ndarray:
    a = np.loadtxt(path, dtype=int)
    return a.ravel() if a.ndim > 1 else a

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def mean_match_loss(h1, h2):
    return torch.mean((h1.mean(dim=0) - h2.mean(dim=0)) ** 2)

def coral_loss(h1, h2, eps: float = 1e-5):
    d = h1.size(1)
    h1c = h1 - h1.mean(dim=0, keepdim=True)
    h2c = h2 - h2.mean(dim=0, keepdim=True)
    c1 = (h1c.t() @ h1c) / max(h1c.size(0) - 1, 1)
    c2 = (h2c.t() @ h2c) / max(h2c.size(0) - 1, 1)
    return ((c1 - c2) ** 2).sum() / (4.0 * d * d + eps)

def _median_sigma(x, y, max_n: int = 512):
    if x.size(0) > max_n:
        x = x[torch.randperm(x.size(0), device=x.device)[:max_n]]
    if y.size(0) > max_n:
        y = y[torch.randperm(y.size(0), device=y.device)[:max_n]]
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z, p=2.0) ** 2
    return torch.sqrt(d2.flatten().median() + 1e-12)

def mmd_loss(h1, h2):
    sigma0 = _median_sigma(h1, h2)
    sigmas = sigma0 * torch.tensor([0.5, 1.0, 2.0, 4.0], device=h1.device)
    dxx = torch.cdist(h1, h1, p=2.0) ** 2
    dyy = torch.cdist(h2, h2, p=2.0) ** 2
    dxy = torch.cdist(h1, h2, p=2.0) ** 2
    Kxx = Kyy = Kxy = 0.0
    for s in sigmas:
        g = 1.0 / (2.0 * s ** 2 + 1e-12)
        Kxx = Kxx + torch.exp(-g * dxx)
        Kyy = Kyy + torch.exp(-g * dyy)
        Kxy = Kxy + torch.exp(-g * dxy)
    B1, B2 = h1.size(0), h2.size(0)
    Kxx = (Kxx.sum() - Kxx.diag().sum()) / max(B1 * (B1 - 1), 1)
    Kyy = (Kyy.sum() - Kyy.diag().sum()) / max(B2 * (B2 - 1), 1)
    Kxy = Kxy.mean()
    return Kxx + Kyy - 2.0 * Kxy

def variance_loss(h, gamma: float = 1.0, eps: float = 1e-4):
    std = torch.sqrt(h.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std) ** 2)

def covariance_loss(h):
    """VICReg-style decorrelation: sum of squared OFF-diagonal covariances / d.
    Pushes Cov(Z) toward diagonal, raising effective rank (Fix 3 / Thm-3 asm (43))."""
    B, d = h.shape
    hc = h - h.mean(dim=0, keepdim=True)
    cov = (hc.t() @ hc) / max(B - 1, 1)          # (d, d)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / d

def _align_triple(a, b):
    return mean_match_loss(a, b), coral_loss(a, b), mmd_loss(a, b)

def conditional_align(h_ptb, h_echo, K: int, min_count: int = 8):
    """Mechanism #1: CONDITIONAL alignment (addresses Assumption-4 covariate shift).

    Instead of forcing the two domains' MARGINAL latent distributions equal
    (which distorts conditionals when covariate/label mix differs across
    datasets), we stratify both domains into K quantile bins of a *shared*
    risk axis (top principal direction of the pooled batch latent) and align
    mean/CORAL/MMD WITHIN each bin, then average. K=1 recovers the original
    unconditional alignment (so it doubles as a clean ablation)."""
    if K <= 1:
        return _align_triple(h_ptb, h_echo)
    with torch.no_grad():
        pooled = torch.cat([h_ptb, h_echo], dim=0)
        mu = pooled.mean(0, keepdim=True)
        # top principal direction of the pooled batch (shared risk axis)
        cov = torch.cov(pooled.t())
        pc = torch.linalg.eigh(cov)[1][:, -1]
        s_ptb  = (h_ptb.detach()  - mu) @ pc
        s_echo = (h_echo.detach() - mu) @ pc
        s_all  = torch.cat([s_ptb, s_echo], dim=0)
        edges = torch.quantile(s_all, torch.linspace(0, 1, K + 1, device=h_ptb.device))
    m_acc = c_acc = d_acc = 0.0
    nb = 0
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        if k == K - 1:
            mp = (s_ptb >= lo) & (s_ptb <= hi); me = (s_echo >= lo) & (s_echo <= hi)
        else:
            mp = (s_ptb >= lo) & (s_ptb < hi);  me = (s_echo >= lo) & (s_echo < hi)
        if mp.sum() >= min_count and me.sum() >= min_count:
            mm, cc, dd = _align_triple(h_ptb[mp], h_echo[me])
            m_acc = m_acc + mm; c_acc = c_acc + cc; d_acc = d_acc + dd; nb += 1
    if nb == 0:                      # degenerate batch -> fall back to unconditional
        return _align_triple(h_ptb, h_echo)
    return m_acc / nb, c_acc / nb, d_acc / nb

def af_anchored_alignment(h_ptb, h_echo):
    """Align Echo toward the current PTB reference without a direct PTB-branch gradient.

    Only the alignment reference is detached. AF supervision, EF supervision,
    and PTB regularization retain their original gradients. Shared parameters
    can still change PTB representations through the Echo branch; this is not
    an AF-risk preservation guarantee or a claim of identified predictive matching.
    The forward loss values equal the original K=1 alignment at fixed inputs.
    """
    return conditional_align(h_ptb.detach(), h_echo, K=1)

class ModalityProjector(nn.Module):
    def __init__(self, in_dim, d, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d), nn.LayerNorm(d),
        )
    def forward(self, x):
        return self.net(x)

class MaskedMultimodalTransformer(nn.Module):
    def __init__(self, dim_ecg=168, dim_meta=2, dim_echo=512, d=128, proj_hidden=256,
                 nhead=4, num_layers=2, ff_dim=256, dropout=0.1, head_hidden=128):
        super().__init__()
        self.proj_ecg  = ModalityProjector(dim_ecg,  d, proj_hidden, dropout)
        self.proj_meta = ModalityProjector(dim_meta, d, proj_hidden, dropout)
        self.proj_echo = ModalityProjector(dim_echo, d, proj_hidden, dropout)
        self.mask_shared = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.modal_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_af = nn.Sequential(nn.Linear(d, head_hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        self.head_ef = nn.Sequential(nn.Linear(d, head_hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        self.d = d

    def _get_token(self, z, proj, modal_vec, B, device, mask_prob=0.0):
        mask_tok = self.mask_shared.expand(B, 1, self.d)
        if z is None:
            return mask_tok
        token = proj(z).unsqueeze(1) + modal_vec.view(1, 1, self.d)
        if mask_prob > 0.0 and self.training:
            rand = torch.rand(B, 1, 1, device=device)
            token = torch.where(rand < mask_prob, mask_tok, token)
        return token

    def forward(self, z_ecg=None, z_meta=None, z_echo=None, mask_prob=0.0, slot_perm=False):
        B, device = self._infer_batch(z_ecg, z_meta, z_echo)
        t_ecg  = self._get_token(z_ecg,  self.proj_ecg,  self.modal_emb[0], B, device, mask_prob)
        t_meta = self._get_token(z_meta, self.proj_meta, self.modal_emb[1], B, device, mask_prob)
        t_echo = self._get_token(z_echo, self.proj_echo, self.modal_emb[2], B, device, mask_prob)
        tokens = torch.cat([t_ecg, t_meta, t_echo], dim=1)
        if slot_perm and self.training:
            perms = torch.stack([torch.randperm(3, device=device) for _ in range(B)], dim=0)
            tokens = tokens.gather(1, perms.unsqueeze(-1).expand(-1, -1, self.d))
        out = self.transformer(tokens)
        h_fused = out.mean(dim=1)
        logit_af = self.head_af(h_fused).squeeze(-1)
        pred_ef  = self.head_ef(h_fused).squeeze(-1)
        return h_fused, logit_af, pred_ef

    @staticmethod
    def _infer_batch(z_ecg, z_meta, z_echo):
        for z in (z_ecg, z_meta, z_echo):
            if z is not None:
                return z.size(0), z.device
        raise ValueError("All modalities are None")

def ptb_val_auc(model, device, Z_ecg, Z_meta, y_af, id_val):
    model.eval()
    xecg  = torch.from_numpy(Z_ecg[id_val]).to(device)
    xmeta = torch.from_numpy(Z_meta[id_val]).to(device)
    logits = []
    for i in range(0, xecg.size(0), 4096):
        _, logit, _ = model(z_ecg=xecg[i:i+4096], z_meta=xmeta[i:i+4096],
                            z_echo=None, mask_prob=0.0, slot_perm=False)
        logits.append(logit.cpu().numpy())
    p = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    yv = y_af[id_val]
    return roc_auc_score(yv, p), average_precision_score(yv, p)
