from pydantic import BaseModel
from typing import List, Dict, Any

class RecommendationItem(BaseModel):
    item_id: int
    title: str
    genres: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    is_cold_start: bool

class HealthResponse(BaseModel):
    status: str
    model: str
    num_items: int
    num_users: int

class MetricsResponse(BaseModel):
    popularity_baseline: Dict[str, float]
    retrieval_only: Dict[str, float]
    full_pipeline: Dict[str, float]
