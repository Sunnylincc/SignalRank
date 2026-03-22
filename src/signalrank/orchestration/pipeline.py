from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from signalrank.common.types import Candidate, UserContext
from signalrank.llm.enrichment import enrich_item
from signalrank.ranking.inference import score_candidates
from signalrank.rerank.policies import diversity_rerank
from signalrank.retrieval.ann import ANNIndex
from signalrank.retrieval.candidate_generation import CandidateGenerator


@dataclass(slots=True)
class PipelineAssets:
    ann: ANNIndex
    candidate_generator: CandidateGenerator
    item_metadata: dict[int, dict[str, str | float]]


def build_assets(items_df: pd.DataFrame, interactions_df: pd.DataFrame, embedding_dim: int = 64) -> PipelineAssets:
    item_ids = items_df["item_id"].astype(int).tolist()
    vectors = []
    metadata: dict[int, dict[str, str | float]] = {}

    for row in items_df.itertuples(index=False):
        enriched = enrich_item(int(row.item_id), str(row.title), str(row.description), embedding_dim=embedding_dim)
        vector = np.asarray(enriched.embedding, dtype=np.float32)
        vectors.append(vector)
        metadata[int(row.item_id)] = {
            "taxonomy": enriched.taxonomy_path[-1],
            "category": str(row.category),
            "language": str(row.language),
            "hours_since_last_seen": 24.0,
        }

    ann = ANNIndex(item_ids=item_ids, vectors=np.vstack(vectors))
    generator = CandidateGenerator.from_interactions(interactions_df)
    return PipelineAssets(ann=ann, candidate_generator=generator, item_metadata=metadata)


def retrieve_candidates(assets: PipelineAssets, user_vector: np.ndarray, user_ctx: UserContext, top_k: int) -> list[Candidate]:
    candidate_pool = assets.candidate_generator.generate(user_ctx.surface, limit=max(500, top_k * 5))
    retrieved = assets.ann.top_k(user_vector, top_k, candidate_ids=candidate_pool)
    return [
        Candidate(item_id=item_id, retrieval_score=score, metadata=assets.item_metadata.get(item_id, {}))
        for item_id, score in retrieved
    ]


def recommend(assets: PipelineAssets, user_vector: np.ndarray, user_ctx: UserContext, top_k: int = 20) -> list[Candidate]:
    candidates = retrieve_candidates(assets, user_vector, user_ctx=user_ctx, top_k=max(top_k * 10, 100))
    ranked = score_candidates(candidates, user_ctx)
    reranked = diversity_rerank(ranked)
    return reranked[:top_k]
