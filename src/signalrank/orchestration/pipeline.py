from dataclasses import dataclass
import numpy as np
from signalrank.common.types import Candidate
from signalrank.retrieval.ann import ANNIndex
from signalrank.rerank.policies import diversity_rerank


@dataclass(slots=True)
class PipelineAssets:
    ann: ANNIndex


def retrieve(ann: ANNIndex, user_vector: np.ndarray, top_k: int) -> list[Candidate]:
    return [Candidate(item_id=i, score=s, metadata={"taxonomy": f"tax_{i % 5}"}) for i, s in ann.top_k(user_vector, top_k)]


def rank(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def recommend(assets: PipelineAssets, user_vector: np.ndarray, top_k_retrieval: int = 200, top_k_final: int = 20) -> list[Candidate]:
    cands = retrieve(assets.ann, user_vector, top_k_retrieval)
    ranked = rank(cands)
    reranked = diversity_rerank(ranked)
    return reranked[:top_k_final]
