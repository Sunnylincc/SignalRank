import numpy as np
from signalrank.evaluation.metrics import auc, binary_logloss, ndcg_at_k, mrr_at_k


def test_metrics_ranges() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.3])
    assert 0.0 <= auc(y, p) <= 1.0
    assert binary_logloss(y, p) >= 0.0
    assert 0.0 <= ndcg_at_k(y.tolist(), p.tolist(), 3) <= 1.0
    assert 0.0 <= mrr_at_k(y.tolist(), p.tolist(), 3) <= 1.0
