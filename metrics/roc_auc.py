from sklearn.metrics import roc_auc_score
import numpy as np

def roc_auc(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy().astype(int)

    if preds.min() >= 0 and preds.max() <= 1:
        probas = preds
    else:
        probas = 1 / (1 + np.exp(-preds))  # Apply sigmoid to logits

    try:
        score = roc_auc_score(targets.flatten(), probas.flatten())
    except ValueError:
        score = 0.0  # e.g., if only one class is present

    fmt = lambda x: f"{x:.4f}"
    agg = lambda scores: sum(scores) / len(scores)
    return score, fmt, agg, "max"
