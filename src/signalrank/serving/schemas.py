from pydantic import BaseModel, Field


class ContextPayload(BaseModel):
    hour: int = Field(default=12, ge=0, le=23)
    surface: str = "home"
    country: str = "US"


class RetrievalRequest(BaseModel):
    user_id: int
    context: ContextPayload = Field(default_factory=ContextPayload)
    top_k: int = Field(default=100, ge=1, le=1000)


class RankRequest(BaseModel):
    user_id: int
    context: ContextPayload = Field(default_factory=ContextPayload)
    candidate_item_ids: list[int]


class RecommendRequest(BaseModel):
    user_id: int
    context: ContextPayload = Field(default_factory=ContextPayload)
    top_k: int = Field(default=20, ge=1, le=100)


class ScoredItem(BaseModel):
    item_id: int
    retrieval_score: float
    rank_score: float
    final_score: float
    taxonomy: str


class RecommendResponse(BaseModel):
    user_id: int
    items: list[ScoredItem]
