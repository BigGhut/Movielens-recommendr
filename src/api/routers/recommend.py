import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import RecommendationResponse, HealthResponse, MetricsResponse
# Для зависимостей можно было бы использовать Depends, но так как у нас один pipeline, мы импортируем его из main
# Чтобы избежать циклических импортов, мы можем получить доступ к app.state
from fastapi import Request

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Модели не загружены")
        
    return HealthResponse(
        status="ok",
        model="HeteroGNN + CatBoost",
        num_items=len(pipeline.idx2item),
        num_users=len(pipeline.user2idx)
    )

@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def recommend(request: Request, user_id: str):
    if '.' in user_id or ',' in user_id:
        raise HTTPException(status_code=400, detail="Ошибка запроса: ID не может быть числом с плавающей точкой.")
    try:
        user_id_int = int(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ошибка запроса: ID должен быть целым числом.")

    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Модели не загружены")
        
    # Cold-start fallback
    is_cold_start = user_id_int not in pipeline.user2idx
    
    # Получаем рекомендации (<= 500ms)
    recs = pipeline.recommend(user_id=user_id_int, top_k=10)
    
    return RecommendationResponse(
        user_id=user_id_int,
        recommendations=recs,
        is_cold_start=is_cold_start
    )

@router.get("/history/{user_id}")
async def get_history(request: Request, user_id: str):
    if '.' in user_id or ',' in user_id:
        raise HTTPException(status_code=400, detail="Ошибка запроса: ID не может быть числом с плавающей точкой.")
    try:
        user_id_int = int(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ошибка запроса: ID должен быть целым числом.")

    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Модели не загружены")
        
    try:
        history = pipeline.get_user_history(user_id_int)
        return {"user_id": user_id_int, "history": history}
    except Exception as e:
        return {"user_id": user_id_int, "history": []}

@router.get("/metrics", response_model=MetricsResponse)
async def metrics():
    # Загружаем метрики из файла, сгенерированного evaluate.py
    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        # Если метрики еще не посчитаны, возвращаем пустые
        return MetricsResponse(
            popularity_baseline={},
            retrieval_only={},
            full_pipeline={}
        )
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    return MetricsResponse(**data)
