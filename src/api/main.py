from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

# Mocking pipeline initialization for robust startup even without trained models
# In a real scenario, models would be loaded here.
class MockPipeline:
    def __init__(self):
        self.user2idx = {}
        self.idx2item = {}
        self.popularity_recommendations = []
        
    def recommend(self, user_id, top_k=10):
        return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Попытка загрузить конфигурацию
    config_path = Path("configs/model_config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
    # В реальности здесь:
    # 1. Загрузка TwoTowerModel
    # 2. Загрузка FAISSIndex
    # 3. Загрузка CatBoostRanker
    # 4. Загрузка Feature Store, словарей user2idx, item2idx
    # 5. Инициализация RecommendationPipeline
    
    try:
        from src.pipeline.recommend import RecommendationPipeline
        from src.retrieval.index import FAISSIndex
        from src.models.ranker import CatBoostRanker
        from src.data.preprocessing import create_id_mappings, load_movies, load_ratings
        from src.data.feature_store import build_user_features, build_item_features, build_user_genre_profiles
        import numpy as np

        print("Загрузка данных...")
        data_dir = Path("data/raw/ml-1m")
        movies_df = load_movies(data_dir)
        ratings_df = load_ratings(data_dir)
        
        users_df = pd.read_csv(data_dir / 'users.dat', sep='::', engine='python', encoding='latin-1',
                               names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        
        user2idx, item2idx = create_id_mappings(ratings_df)
        idx2user = {v: k for k, v in user2idx.items()}
        idx2item = {v: k for k, v in item2idx.items()}
        
        print("Загрузка фичей...")
        user_features = build_user_features(ratings_df, movies_df, users_df)
        item_features = build_item_features(ratings_df, movies_df)
        user_genre_profiles = build_user_genre_profiles(ratings_df, movies_df)
        
        print("Загрузка моделей...")
        faiss_index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
        catboost_ranker = CatBoostRanker.load(Path("artifacts/models/catboost_ranker.cbm"))
        
        user_embs = np.load("artifacts/models/user_embeddings.npy")
        item_embs = np.load("artifacts/models/item_embeddings.npy")
        
        print("Инициализация пайплайна...")
        pipeline = RecommendationPipeline(
            user_embs=user_embs, item_embs=item_embs,
            faiss_index=faiss_index, catboost_ranker=catboost_ranker,
            user_features=user_features, item_features=item_features,
            user2idx=user2idx, item2idx=item2idx, idx2item=idx2item,
            movies_df=movies_df, ratings_df=ratings_df, config=config, user_genre_profiles=user_genre_profiles
        )
        app.state.pipeline = pipeline
        print("Модели успешно загружены!")
        
    except Exception as e:
        print(f"Внимание: Ошибка при загрузке полных моделей ({e}). Запуск в мок-режиме.")
        app.state.pipeline = MockPipeline()
        
    yield
    # Очистка ресурсов, если нужно

app = FastAPI(
    title="Movie Recommender API",
    description="HeteroGNN + CatBoost Recommender System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routers import recommend
app.include_router(recommend.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
