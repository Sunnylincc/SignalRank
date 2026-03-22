from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Candidate:
    item_id: int
    score: float
    metadata: dict[str, Any] | None = None
