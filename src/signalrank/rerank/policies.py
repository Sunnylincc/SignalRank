from collections import defaultdict
from signalrank.common.types import Candidate


def diversity_rerank(candidates: list[Candidate], max_per_taxonomy: int = 3) -> list[Candidate]:
    buckets: dict[str, int] = defaultdict(int)
    reranked: list[Candidate] = []

    for candidate in sorted(candidates, key=lambda x: x.rank_score, reverse=True):
        taxonomy = str(candidate.metadata.get("taxonomy", "unknown"))
        if buckets[taxonomy] >= max_per_taxonomy:
            continue
        buckets[taxonomy] += 1
        candidate.final_score = candidate.rank_score - 0.01 * buckets[taxonomy]
        reranked.append(candidate)

    return sorted(reranked, key=lambda x: x.final_score, reverse=True)
