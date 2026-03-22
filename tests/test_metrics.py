import numpy as np

from signalrank.evaluation.metrics import auc, binary_logloss, mrr_at_k, ndcg_at_k
from signalrank.retrieval.ann import ANNIndex


def test_metrics_ranges() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.3])
    assert 0.0 <= auc(y, p) <= 1.0
    assert binary_logloss(y, p) >= 0.0
    assert 0.0 <= ndcg_at_k(y.tolist(), p.tolist(), 3) <= 1.0
    assert 0.0 <= mrr_at_k(y.tolist(), p.tolist(), 3) <= 1.0


def test_ann_subset_topk() -> None:
    index = ANNIndex(item_ids=[1, 2, 3], vectors=np.array([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]], dtype=np.float32))
    got = index.top_k(np.array([1.0, 0.0], dtype=np.float32), k=2, candidate_ids=[2, 3])
    assert [item_id for item_id, _ in got] == [2, 3]
