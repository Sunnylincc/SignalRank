import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred))


def binary_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_pred))


def ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    ranked = np.array(labels)[order]
    dcg = np.sum((2**ranked - 1) / np.log2(np.arange(2, len(ranked) + 2)))
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(labels: list[int], scores: list[float], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0:
            return 1.0 / rank
    return 0.0
