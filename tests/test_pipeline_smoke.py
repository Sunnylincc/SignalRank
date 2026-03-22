import numpy as np
from signalrank.orchestration.pipeline import PipelineAssets, recommend
from signalrank.retrieval.ann import ANNIndex


def test_recommend_smoke() -> None:
    ids = [1, 2, 3, 4]
    vecs = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    assets = PipelineAssets(ann=ANNIndex(ids, vecs))
    recs = recommend(assets, np.array([1.0, 0.0], dtype=np.float32), top_k_retrieval=4, top_k_final=2)
    assert len(recs) == 2
    assert recs[0].item_id in ids
