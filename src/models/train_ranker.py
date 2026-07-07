import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from src.data.preprocessing import load_ratings, load_movies, load_users, temporal_split, create_id_mappings
from src.data.feature_store import build_user_features, build_item_features, build_pair_features, build_user_genre_profiles
from src.models.ranker import CatBoostRanker
from src.models.two_tower import TwoTowerModel
from src.retrieval.index import FAISSIndex
from src.retrieval.embeddings import generate_all_user_embeddings, generate_all_item_embeddings
import torch

def prepare_ranker_data(df: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame, 
                        user_embs: np.ndarray, item_embs: np.ndarray, 
                        user2idx: dict, item2idx: dict, idx2item: dict, 
                        faiss_index: FAISSIndex, user_genre_profiles: dict, top_k: int = 200, positive_threshold: float = 4.0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Подготавливает датасет для ранжирования.
    Для каждого юзера берет топ-K кандидатов из FAISS.
    Позитивные - те, что юзер оценил >= positive_threshold.
    Негативные - кандидаты из топ-K без оценок.
    """
    user_item_ratings = df.set_index(['user_id', 'item_id'])['rating'].to_dict()
    all_users = df['user_id'].unique()
    
    pairs = []
    labels = []
    
    for u_id in tqdm(all_users, desc="Preparing ranking data"):
        if u_id not in user2idx:
            continue
            
        u_idx = user2idx[u_id]
        u_emb = user_embs[u_idx:u_idx+1]
        
        # Получаем кандидатов
        distances, indices = faiss_index.search(u_emb, top_k)
        candidates = indices[0]
        
        for rank, c_idx in enumerate(candidates):
            if c_idx not in idx2item:
                continue
            i_id = idx2item[c_idx]
            
            # Проверяем, есть ли реальный рейтинг в этом сплите
            rating = user_item_ratings.get((u_id, i_id), 0.0)
            
            # Label Smoothing (v4)
            epsilon = 0.15 # коэффициент смягчения
            s_gnn = np.dot(u_emb[0], item_embs[c_idx])
            # Клиппинг для стабильности экспоненты
            s_gnn_clipped = np.clip(s_gnn, -10, 10)
            prob_gnn = 1.0 / (1.0 + np.exp(-s_gnn_clipped))
            
            # Смягчение: смешиваем жесткий рейтинг и предсказание графа
            label = (1 - epsilon) * rating + epsilon * 5.0 * prob_gnn
                
            pairs.append({'user_id': u_id, 'item_id': i_id, 'retrieval_rank': float(rank)})
            labels.append(label)
            
    pairs_df = pd.DataFrame(pairs)
    labels = np.array(labels)
    
    # Собираем фичи
    X = build_pair_features(user_features, item_features, user_embs, item_embs, user2idx, item2idx, pairs_df, user_genre_profiles)
    
    
    # Сохраняем user_id как qid
    qids = pairs_df['user_id'].values
    
    # Убираем id колонки
    X = X.drop(columns=['user_id', 'item_id', 'title'], errors='ignore')
    return X, labels, qids

def train_ranker(config: dict, train_df, val_df, user_features, item_features, 
                 user_embs, item_embs, user2idx, item2idx, idx2item, faiss_index, user_genre_profiles, save_dir: Path) -> CatBoostRanker:
    
    print("Подготовка обучающей выборки для Ranker...")
    X_train, y_train, qid_train = prepare_ranker_data(train_df, user_features, item_features, user_embs, item_embs, 
                                           user2idx, item2idx, idx2item, faiss_index, user_genre_profiles, top_k=config.get('retrieval_top_k', 200), positive_threshold=config['positive_threshold'])
                                           
    print("Подготовка валидационной выборки для Ranker...")
    X_val, y_val, qid_val = prepare_ranker_data(val_df, user_features, item_features, user_embs, item_embs, 
                                       user2idx, item2idx, idx2item, faiss_index, user_genre_profiles, top_k=config.get('retrieval_top_k', 200), positive_threshold=config['positive_threshold'])
    
    # Extract baseline
    baseline_train = X_train['cosine_similarity'].values
    baseline_val = X_val['cosine_similarity'].values
    X_train = X_train.drop(columns=['cosine_similarity'])
    X_val = X_val.drop(columns=['cosine_similarity'])
    
    # Категориальные фичи (из feature_store.py)
    cat_features = ['most_common_genre', 'gender', 'occupation', 'zip']
    text_features = ['genres']
    
    # Заполняем пропуски в категориальных фичах
    for col in cat_features + text_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna("Unknown").astype(str)
            X_val[col] = X_val[col].fillna("Unknown").astype(str)
            
    # Подготавливаем текстовые фичи для токенизации (CatBoost по умолчанию бьет по пробелам)
    for col in text_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].str.replace('|', ' ', regex=False)
            X_val[col] = X_val[col].str.replace('|', ' ', regex=False)
            
    print("Обучение CatBoost...")
    ranker = CatBoostRanker(config)
    
    ranker.fit(X_train, y_train, qid_train, X_val, y_val, qid_val, 
               baseline_train=baseline_train, baseline_val=baseline_val,
               cat_features=cat_features, text_features=text_features)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(save_dir / "catboost_ranker.cbm")
    print(f"Модель сохранена в {save_dir / 'catboost_ranker.cbm'}")
    
    return ranker

if __name__ == '__main__':
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)["ranker"]
        
    print("Загрузка данных для CatBoost...")
    data_dir = Path("data/raw/ml-1m")
    ratings = load_ratings(data_dir)
    movies_df = load_movies(data_dir)
    users_df = load_users(data_dir)
    
    train_df, val_df, test_df = temporal_split(ratings)
    user2idx, item2idx = create_id_mappings(train_df)
    idx2item = {v: k for k, v in item2idx.items()}
    
    print("Генерация признаков...")
    user_features = build_user_features(ratings, movies_df, users_df)
    item_features = build_item_features(ratings, movies_df)
    
    user_embs = np.load("artifacts/models/user_embeddings.npy")
    item_embs = np.load("artifacts/models/item_embeddings.npy")
    
    faiss_index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
    
    user_genre_profiles = build_user_genre_profiles(ratings, movies_df)
    
    train_ranker(
        config, train_df, val_df, 
        user_features, item_features, 
        user_embs, item_embs, 
        user2idx, item2idx, idx2item, 
        faiss_index, user_genre_profiles, Path("artifacts/models")
    )
