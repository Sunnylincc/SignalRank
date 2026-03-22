from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class EnrichedItem:
    item_id: int
    semantic_tags: list[str]
    taxonomy_path: list[str]
    embedding: list[float]


def enrich_item(item_id: int, title: str, description: str, embedding_dim: int = 16) -> EnrichedItem:
    text = f"{title} {description}".lower()
    tags = [t for t in ["sports", "finance", "fashion", "electronics", "travel"] if t in text]
    if not tags:
        tags = ["general"]
    taxonomy = ["root", tags[0]]
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    emb = rng.normal(0, 1, embedding_dim).astype(float).tolist()
    return EnrichedItem(item_id=item_id, semantic_tags=tags, taxonomy_path=taxonomy, embedding=emb)
