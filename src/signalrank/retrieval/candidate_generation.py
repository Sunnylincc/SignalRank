from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class CandidateGenerator:
    top_items_global: list[int]
    top_items_by_surface: dict[str, list[int]]

    @classmethod
    def from_interactions(cls, interactions: pd.DataFrame, top_n: int = 500) -> "CandidateGenerator":
        clicks = interactions[interactions["event_type"].isin(["click", "conversion"])]
        global_counts = Counter(clicks["item_id"].tolist())
        top_global = [item for item, _ in global_counts.most_common(top_n)]

        by_surface: dict[str, Counter[int]] = defaultdict(Counter)
        for row in clicks.itertuples(index=False):
            by_surface[row.surface][row.item_id] += 1

        top_by_surface = {
            surface: [item for item, _ in counter.most_common(top_n)] for surface, counter in by_surface.items()
        }
        return cls(top_items_global=top_global, top_items_by_surface=top_by_surface)

    def generate(self, surface: str, limit: int = 1000) -> list[int]:
        pool = self.top_items_by_surface.get(surface, self.top_items_global)
        return pool[:limit]
