from sklearn.metrics import f1_score as sk_f1_score
import numpy as np
import torch

def f1_score(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Flatten to [N*S] and ensure binary integers
    preds_flat = preds.astype(np.int32).flatten()
    targets_flat = targets.astype(np.int32).flatten()

    # Compute F1 score using micro averaging
    score = sk_f1_score(targets_flat, preds_flat, average='micro')

    fmt = lambda x: f"{x:.4f}"
    agg = lambda scores: sum(scores) / len(scores)
    return score, fmt, agg, "max"
