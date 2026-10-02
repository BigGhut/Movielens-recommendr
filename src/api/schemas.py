from pydantic import BaseModel


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    genres: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    is_cold_start: bool

class HealthResponse(BaseModel):
    status: str
    model: str
    num_items: int
    num_users: int

class MetricsResponse(BaseModel):
    popularity_baseline: dict[str, float]
    retrieval_only: dict[str, float]
    full_pipeline: dict[str, float]
