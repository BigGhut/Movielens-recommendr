import numpy as np
import pandas as pd
from src.retrieval.index import FAISSIndex
from src.models.ranker import CatBoostRanker
from src.data.feature_store import build_pair_features

class RecommendationPipeline:
    def __init__(self, user_embs: np.ndarray, item_embs: np.ndarray, 
                 faiss_index: FAISSIndex, catboost_ranker: CatBoostRanker, 
                 user_features: pd.DataFrame, item_features: pd.DataFrame, 
                 user2idx: dict, item2idx: dict, idx2item: dict, 
                 movies_df: pd.DataFrame, ratings_df: pd.DataFrame, config: dict, user_genre_profiles: dict = None):
        self.user_embs = user_embs
        self.item_embs = item_embs
        self.faiss_index = faiss_index
        self.catboost_ranker = catboost_ranker
        self.user_features = user_features
        self.item_features = item_features
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.movies_df = movies_df
        
        # Для фильтрации уже просмотренного
        self.user_history = ratings_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.config = config
        self.user_genre_profiles = user_genre_profiles
        
        # Предрасчет популярных фильмов для cold-start
        item_counts = ratings_df['item_id'].value_counts()
        top_items = item_counts.head(100).index.tolist()
        self.popularity_recommendations = []
        for item_id in top_items:
            movie_info = self.movies_df[self.movies_df['item_id'] == item_id]
            if not movie_info.empty:
                self.popularity_recommendations.append({
                    'item_id': item_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'score': float(item_counts[item_id])
                })

    def recommend(self, user_id: int, top_k: int = 10) -> list[dict]:
        """
        Full pipeline: Retrieval -> Re-ranking
        Возвращает [{item_id, title, genres, score}]
        """
        if user_id not in self.user2idx:
            # Fallback для новых пользователей
            return self.recommend_cold_start(top_k)
            
        u_idx = self.user2idx[user_id]
        u_emb = self.user_embs[u_idx:u_idx+1]
        
        # 1. Retrieval
        retrieval_k = self.config.get('retrieval_top_k', 200)
        distances, indices = self.faiss_index.search(u_emb, retrieval_k * 2) # берем с запасом для фильтрации истории
        
        candidates = []
        history = self.user_history.get(user_id, set())
        
        for c_idx in indices[0]:
            if c_idx not in self.idx2item:
                continue
            item_id = self.idx2item[c_idx]
            if item_id not in history:
                candidates.append(item_id)
            if len(candidates) >= retrieval_k:
                break
                
        if not candidates:
            return self.recommend_cold_start(top_k)
            
        # 2. Re-ranking
        ranks = list(range(len(candidates)))
        pairs_df = pd.DataFrame({'user_id': [user_id]*len(candidates), 'item_id': candidates, 'retrieval_rank': [0.0]*len(candidates)})
        
        X = build_pair_features(self.user_features, self.item_features, 
                                self.user_embs, self.item_embs, 
                                self.user2idx, self.item2idx, pairs_df, self.user_genre_profiles)
                                
        cat_features = ['most_common_genre', 'gender', 'occupation', 'zip']
        text_features = ['genres']
        for col in cat_features + text_features:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
                
        for col in text_features:
            if col in X.columns:
                X[col] = X[col].str.replace('|', ' ', regex=False)
                
        X_predict = X.drop(columns=['user_id', 'item_id', 'title'], errors='ignore')
        
        baseline = X_predict['cosine_similarity'].values
        X_predict = X_predict.drop(columns=['cosine_similarity'], errors='ignore')
        
        scores = self.catboost_ranker.predict(X_predict)
        
        # Сортировка по скору CatBoost + baseline
        pairs_df['score'] = scores + baseline
        pairs_df = pairs_df.sort_values('score', ascending=False).head(top_k)
        
        # Формирование ответа
        results = []
        for _, row in pairs_df.iterrows():
            item_id = int(row['item_id'])
            movie_info = self.movies_df[self.movies_df['item_id'] == item_id]
            if not movie_info.empty:
                results.append({
                    'item_id': item_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'score': float(row['score'])
                })
                
        return results

    def recommend_cold_start(self, top_k: int = 10) -> list[dict]:
        """Возвращает популярные фильмы для холодных пользователей."""
        return self.popularity_recommendations[:top_k]
        
    def get_user_history(self, user_id: int) -> list[dict]:
        """Возвращает историю просмотров юзера."""
        history_ids = self.user_history.get(user_id, set())
        results = []
        for item_id in history_ids:
            movie_info = self.movies_df[self.movies_df['item_id'] == item_id]
            if not movie_info.empty:
                results.append({
                    'item_id': int(item_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres']
                })
        return results
