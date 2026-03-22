from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=100, ge=1, le=1000)


class RankRequest(BaseModel):
    user_id: int
    candidate_item_ids: list[int]


class RecommendRequest(BaseModel):
    user_id: int
    context: dict[str, str | int | float] = Field(default_factory=dict)
    top_k: int = Field(default=20, ge=1, le=100)


class ScoredItem(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    items: list[ScoredItem]
