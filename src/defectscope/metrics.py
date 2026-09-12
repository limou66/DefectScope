"""Small transparent implementations, including correct average tie ranks."""
import math
import numpy as np

def auroc(labels, scores):
    y, s = np.asarray(labels, dtype=bool).ravel(), np.asarray(scores, dtype=float).ravel()
    if len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError("invalid labels/scores")
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))

def conformal_threshold(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=float))
    if not len(s) or not 0 < alpha < 1 or not np.isfinite(s).all():
        raise ValueError("finite nonempty scores and 0<alpha<1 required")
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k-1]) if k <= len(s) else float("inf")

def normal_p_value(calibration_scores, score):
    scores = np.asarray(calibration_scores)
    return float((1 + np.count_nonzero(scores >= score)) / (1 + len(scores)))

def binary_metrics(labels, predictions):
    y, p = np.asarray(labels, dtype=bool), np.asarray(predictions, dtype=bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp+fp) if tp+fp else 0.0,
            "recall": tp / (tp+fn) if tp+fn else 0.0,
            "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "fpr": fp / (fp+tn) if fp+tn else None}
