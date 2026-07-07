import argparse
import asyncio
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Mock Two-Tower RecSys API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    fallback: bool
    fallback_type: str = None

@app.get("/health")
async def health(delay: float = Query(0.0, description="Simulated delay in seconds")):
    if delay > 0:
        await asyncio.sleep(delay)
    return {"status": "healthy"}

_train_users_cache = set()
_train_users_mtime = 0.0

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def recommend(
    user_id: int, 
    delay: float = Query(0.0, description="Simulated delay in seconds")
):
    global _train_users_cache, _train_users_mtime
    # Simulate latency
    if delay > 0:
        await asyncio.sleep(delay)
        
    # Check if user is in training set (if train.csv exists)
    # If not, treat as cold user fallback
    is_cold = False
    train_csv = "data/processed/train.csv"
    data_dir_env = os.getenv("DATA_DIR")
    if data_dir_env:
        train_csv = os.path.join(data_dir_env, "train.csv")
        
    if os.path.exists(train_csv):
        try:
            mtime = os.path.getmtime(train_csv)
            if mtime > _train_users_mtime:
                train_df = pd.read_csv(train_csv)
                _train_users_cache = set(train_df["user_id"].unique())
                _train_users_mtime = mtime
            if user_id not in _train_users_cache:
                is_cold = True
        except Exception:
            pass
            
    # Check if index files exist to simulate DB failure graceful degradation
    index_file = "models/item_index.faiss"
    reranker_file = "models/reranker.lgb"
    model_dir_env = os.getenv("MODEL_DIR")
    if model_dir_env:
        index_file = os.path.join(model_dir_env, "item_index.faiss")
        reranker_file = os.path.join(model_dir_env, "reranker.lgb")
        
    if not os.path.exists(index_file) or not os.path.exists(reranker_file):
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
            fallback=True,
            fallback_type="popularity_fallback"
        )

    if user_id >= 900000 or is_cold:
        # Cold start user using textual fallback
        fallback_recs = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        return RecommendationResponse(
            user_id=user_id,
            recommendations=fallback_recs,
            fallback=True,
            fallback_type="textual_description"
        )
        
    if user_id < 0:
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
            fallback=True,
            fallback_type="popularity_fallback"
        )

    # Regular user: re-ranking of 200 candidates to top 10
    candidates = list(range(1, 201))
    
    # Deterministic mock re-ranking scoring
    def score_fn(movie_id):
        return (movie_id * 17 + user_id * 13) % 97
    
    scored_candidates = sorted(candidates, key=score_fn, reverse=True)
    top_10 = scored_candidates[:10]
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=top_10,
        fallback=False
    )

@app.get("/movie/{movie_id}")
async def get_movie(movie_id: int):
    # Cold start for invalid items (returning HTTP 404)
    if movie_id >= 900000 or movie_id < 0:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"movie_id": movie_id, "title": f"Mock Movie {movie_id}", "genres": "Action|Comedy"}

@app.get("/item/{item_id}")
async def get_item(item_id: int):
    if item_id >= 900000 or item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "title": f"Mock Item {item_id}", "genres": "Action|Comedy"}

class RecommendationRequest(BaseModel):
    user_id: int
    num_recs: int = 10

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_post(
    req: RecommendationRequest,
    delay: float = Query(0.0, description="Simulated delay in seconds")
):
    return await recommend(req.user_id, delay)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
