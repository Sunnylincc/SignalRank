from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI

from signalrank.common.types import UserContext
from signalrank.orchestration.pipeline import build_assets, recommend, retrieve_candidates
from signalrank.ranking.inference import score_candidates
from signalrank.serving.schemas import RankRequest, RecommendRequest, RecommendResponse, RetrievalRequest, ScoredItem


app = FastAPI(title="SignalRank API", version="1.0.0")

_DATA_DIR = Path("data/sample")
_items_df = pd.read_csv(_DATA_DIR / "items.csv")
_interactions_df = pd.read_csv(_DATA_DIR / "interactions.csv")
_assets = build_assets(_items_df, _interactions_df)


def _user_vector(user_id: int, dim: int = 64) -> np.ndarray:
    return np.random.default_rng(user_id).normal(size=(dim,)).astype(np.float32)


def _ctx(user_id: int, ctx: RetrievalRequest | RankRequest | RecommendRequest) -> UserContext:
    return UserContext(user_id=user_id, hour=ctx.context.hour, surface=ctx.context.surface, country=ctx.context.country)


def _to_scored_item(c) -> ScoredItem:
    return ScoredItem(
        item_id=c.item_id,
        retrieval_score=float(c.retrieval_score),
        rank_score=float(c.rank_score),
        final_score=float(c.final_score),
        taxonomy=str(c.metadata.get("taxonomy", "unknown")),
    )


@app.post("/retrieve", response_model=list[ScoredItem])
def retrieve_endpoint(req: RetrievalRequest) -> list[ScoredItem]:
    candidates = retrieve_candidates(_assets, _user_vector(req.user_id), _ctx(req.user_id, req), top_k=req.top_k)
    return [_to_scored_item(c) for c in candidates]


@app.post("/rank", response_model=list[ScoredItem])
def rank_endpoint(req: RankRequest) -> list[ScoredItem]:
    user_ctx = _ctx(req.user_id, req)
    lookup = set(req.candidate_item_ids)
    candidates = retrieve_candidates(_assets, _user_vector(req.user_id), user_ctx, top_k=max(len(lookup), 1) * 2)
    filtered = [c for c in candidates if c.item_id in lookup]
    ranked = score_candidates(filtered, user_ctx)
    return [_to_scored_item(c) for c in ranked]


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    user_ctx = _ctx(req.user_id, req)
    recs = recommend(_assets, _user_vector(req.user_id), user_ctx, top_k=req.top_k)
    return RecommendResponse(user_id=req.user_id, items=[_to_scored_item(c) for c in recs])
