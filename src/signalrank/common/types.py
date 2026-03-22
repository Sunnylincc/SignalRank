from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Candidate:
    item_id: int
    retrieval_score: float
    rank_score: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserContext:
    user_id: int
    hour: int = 12
    surface: str = "home"
    country: str = "US"
