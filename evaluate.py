import argparse
import json
import yaml
from pathlib import Path
from src.data.preprocessing import load_ratings, temporal_split
from src.evaluation.metrics import evaluate_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--data-dir", default="data/raw/ml-1m")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("Загрузка данных...")
    ratings = load_ratings(Path(args.data_dir))
    train_df, val_df, test_df = temporal_split(ratings)
    
    # Сбор ground truth (оценки >= 4 считаем позитивными для метрик)
    positive_threshold = config['ranker']['positive_threshold']
    test_positive = test_df[test_df['rating'] >= positive_threshold]
    ground_truth = test_positive.groupby('user_id')['item_id'].apply(set).to_dict()
    
    from tqdm import tqdm
    from src.pipeline.recommend import RecommendationPipeline
    from src.retrieval.candidates import unseen_candidates
    from src.retrieval.index import FAISSIndex
    from src.models.ranker import CatBoostRanker
    from src.data.preprocessing import create_id_mappings, load_users, load_movies
    from src.data.feature_store import (
        build_item_features, build_recent_centroids, build_user_features, build_user_genre_profiles,
    )
    import numpy as np
    
    users_df = load_users(Path(args.data_dir))
    movies_df = load_movies(Path(args.data_dir))
    user2idx, item2idx = create_id_mappings(train_df)
    idx2item = {v: k for k, v in item2idx.items()}
    
    user_features = build_user_features(train_df, movies_df, users_df)
    item_features = build_item_features(train_df, movies_df)
    user_genre_profiles = build_user_genre_profiles(train_df, movies_df)

    faiss_index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
    catboost_ranker = CatBoostRanker.load(Path("artifacts/models/catboost_ranker.cbm"))
    user_embs = np.load("artifacts/models/user_embeddings.npy")
    item_embs = np.load("artifacts/models/item_embeddings.npy")
    recent_k = int(config.get("two_tower", {}).get("recent_k", 10))
    user_recent_embs = build_recent_centroids(train_df, item_embs, user2idx, item2idx, recent_k)
    
    pipeline = RecommendationPipeline(
        user_embs=user_embs, item_embs=item_embs,
        faiss_index=faiss_index, catboost_ranker=catboost_ranker,
        user_features=user_features, item_features=item_features,
        user2idx=user2idx, item2idx=item2idx, idx2item=idx2item,
        movies_df=movies_df, ratings_df=train_df, config=config, user_genre_profiles=user_genre_profiles,
        user_recent_embs=user_recent_embs,
    )
    
    pop_recs = {}
    ret_recs = {}
    full_recs = {}
    
    pop_items = [x['item_id'] for x in pipeline.recommend_cold_start(10)]
    
    for u_id in tqdm(ground_truth.keys(), desc="Evaluating"):
        pop_recs[u_id] = pop_items
        
        if u_id not in pipeline.user2idx:
            ret_recs[u_id] = pop_items
            full_recs[u_id] = pop_items
            continue
            
        full_result = pipeline.recommend(u_id, top_k=10)
        full_recs[u_id] = [x['item_id'] for x in full_result]
        
        u_idx = pipeline.user2idx[u_id]
        u_emb = pipeline.user_embs[u_idx:u_idx+1]
        history = pipeline.user_history.get(u_id, set())
        # Same candidate list the ranker reorders. Retrieval-only keeps FAISS order.
        ret_recs[u_id] = unseen_candidates(
            u_emb, pipeline.faiss_index, pipeline.idx2item, history, pipeline._retrieval_k()
        )[:10]
        
    results = {
        "popularity_baseline": evaluate_model(pop_recs, ground_truth, 10),
        "retrieval_only": evaluate_model(ret_recs, ground_truth, 10),
        "full_pipeline": evaluate_model(full_recs, ground_truth, 10)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
