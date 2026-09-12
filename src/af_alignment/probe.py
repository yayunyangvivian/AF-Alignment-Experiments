"""Validation-only selection score for the downstream frozen logistic probe."""
import warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

@torch.no_grad()
def export_latents(model, ecg, metadata, indices, batch_size=1024):
    previous_mode = model.training
    model.eval()
    try:
        chunks = []
        for start in range(0, len(indices), batch_size):
            ii = indices[start:start + batch_size]
            chunks.append(model(z_ecg=torch.from_numpy(ecg[ii]),
                                z_meta=torch.from_numpy(metadata[ii]))[0].numpy())
        return np.concatenate(chunks)
    finally:
        model.train(previous_mode)

def fit_probe(train_latents, train_labels):
    normalized = train_latents / (np.linalg.norm(train_latents, axis=1, keepdims=True) + 1e-8)
    scaler = StandardScaler().fit(normalized)
    probe = LogisticRegression(max_iter=5000, class_weight='balanced')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        probe.fit(scaler.transform(normalized), train_labels)
    return scaler, probe

def predict_probe(scaler, probe, latents):
    normalized = latents / (np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8)
    return probe.predict_proba(scaler.transform(normalized))[:, 1]

def validation_probe_auc(model, ecg, metadata, train_ids, validation_ids,
                         train_labels, validation_labels):
    """Fit only on training labels, score only validation; preserve training RNG/mode.

    No test labels, indices, or probabilities enter this interface. The scalar
    is a selection criterion, not an unbiased estimate after epoch selection.
    """
    torch_rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()
    previous_mode = model.training
    try:
        ht = export_latents(model, ecg, metadata, train_ids)
        hv = export_latents(model, ecg, metadata, validation_ids)
        scaler, probe = fit_probe(ht, train_labels)
        return float(roc_auc_score(validation_labels, predict_probe(scaler, probe, hv)))
    finally:
        torch.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        model.train(previous_mode)
