from collections import defaultdict
from signalrank.common.types import Candidate


def diversity_rerank(candidates: list[Candidate], max_per_taxonomy: int = 3) -> list[Candidate]:
    buckets: dict[str, int] = defaultdict(int)
    output: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.score, reverse=True):
        tax = (c.metadata or {}).get("taxonomy", "unknown")
        if buckets[tax] >= max_per_taxonomy:
            continue
        buckets[tax] += 1
        output.append(c)
    return output
