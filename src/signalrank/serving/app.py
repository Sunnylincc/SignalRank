from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from signalrank.orchestration.pipeline import PipelineAssets, recommend, retrieve, rank
from signalrank.retrieval.ann import ANNIndex
from signalrank.serving.schemas import RankRequest, RecommendRequest, RecommendResponse, RetrievalRequest, ScoredItem


app = FastAPI(title="SignalRank API", version="0.1.0")

_item_ids = list(range(1000, 2000))
_vectors = np.random.default_rng(7).normal(size=(len(_item_ids), 64)).astype(np.float32)
_assets = PipelineAssets(ann=ANNIndex(_item_ids, _vectors))


def _user_vector(user_id: int, dim: int = 64) -> np.ndarray:
    return np.random.default_rng(user_id).normal(size=(dim,)).astype(np.float32)


@app.post("/retrieve", response_model=list[ScoredItem])
def retrieve_endpoint(req: RetrievalRequest) -> list[ScoredItem]:
    cands = retrieve(_assets.ann, _user_vector(req.user_id), req.top_k)
    return [ScoredItem(item_id=c.item_id, score=c.score) for c in cands]


@app.post("/rank", response_model=list[ScoredItem])
def rank_endpoint(req: RankRequest) -> list[ScoredItem]:
    seed = _user_vector(req.user_id)
    base = float(seed.mean())
    cands = [
        type("Tmp", (), {"item_id": item_id, "score": base + (item_id % 13) * 0.01, "metadata": {"taxonomy": f"tax_{item_id % 4}"}})()
        for item_id in req.candidate_item_ids
    ]
    ranked = rank(cands)
    return [ScoredItem(item_id=c.item_id, score=c.score) for c in ranked]


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    recs = recommend(_assets, _user_vector(req.user_id), top_k_retrieval=max(100, req.top_k * 10), top_k_final=req.top_k)
    return RecommendResponse(user_id=req.user_id, items=[ScoredItem(item_id=r.item_id, score=r.score) for r in recs])
