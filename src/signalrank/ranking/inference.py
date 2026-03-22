from __future__ import annotations

import math
from signalrank.common.types import Candidate, UserContext


def score_candidates(candidates: list[Candidate], user_ctx: UserContext) -> list[Candidate]:
    """Deterministic lightweight rank scoring proxy for online inference.

    Combines retrieval score, recency prior, and surface/taxonomy match boosts.
    """
    scored: list[Candidate] = []
    for c in candidates:
        recency_hours = float(c.metadata.get("hours_since_last_seen", 48.0))
        taxonomy = str(c.metadata.get("taxonomy", "unknown"))

        freshness = math.exp(-recency_hours / 72.0)
        surface_boost = 0.05 if user_ctx.surface in ("home", "search") else 0.0
        taxonomy_boost = 0.04 if taxonomy != "unknown" else 0.0

        c.rank_score = 0.75 * c.retrieval_score + 0.20 * freshness + surface_boost + taxonomy_boost
        c.final_score = c.rank_score
        scored.append(c)

    return sorted(scored, key=lambda x: x.rank_score, reverse=True)
