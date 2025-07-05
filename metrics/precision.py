from sklearn.metrics import precision_score
import numpy as np

def precision(preds, targets):
    preds = preds.cpu().numpy().astype(int)
    targets = targets.cpu().numpy().astype(int)
    score = precision_score(targets.flatten(), preds.flatten(), zero_division=0)
    fmt = lambda x: f"{x:.4f}"
    agg = lambda scores: sum(scores) / len(scores)
    return score, fmt, agg, "max"
