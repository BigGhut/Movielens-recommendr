import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.feature_store import build_pair_features
from src.models.ranker import CatBoostRanker, ranker_inputs
from src.retrieval.candidates import unseen_candidates
from src.retrieval.index import FAISSIndex
from src.retrieval.rp3 import (
    TrainRP3,
    rp3_enabled_settings,
    source_feature_columns,
    trained_source_columns,
    two_tower_score_row,
    union_with_rp3,
)

BLEND_PATH = Path("artifacts/models/ranker_blend.json")


def _build_rp3(
    config: dict,
    train_df: pd.DataFrame | None,
    user2idx: dict,
    item2idx: dict,
) -> TrainRP3 | None:
    """Train-only RP3 hop. The flag off returns None before any graph is built."""
    settings = rp3_enabled_settings(config)
    if settings is None:
        return None
    if train_df is None:
        raise ValueError(
            "retrieval.rp3.enabled requires train_df so validation and test stay out of the graph"
        )
    return TrainRP3.from_frame(
        train_df,
        user2idx,
        item2idx,
        settings["alpha"],
        settings["beta"],
        settings["top_k"],
    )


def _load_ranker_blend(path: Path = BLEND_PATH) -> tuple[str, float]:
    """How to mix the ranker score with cosine. Missing file means model score only."""
    if not path.exists():
        return "blend", 0.0
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    mode = payload.get("mode", "blend")
    beta = float(payload.get("beta", 0.0))
    return mode, beta


class RecommendationPipeline:
    def __init__(self, user_embs: np.ndarray, item_embs: np.ndarray, 
                 faiss_index: FAISSIndex, catboost_ranker: CatBoostRanker, 
                 user_features: pd.DataFrame, item_features: pd.DataFrame, 
                 user2idx: dict, item2idx: dict, idx2item: dict, 
                 movies_df: pd.DataFrame, ratings_df: pd.DataFrame, config: dict, user_genre_profiles: dict | None = None,
                 user_recent_embs: np.ndarray | None = None, train_df: pd.DataFrame | None = None):
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
        self.user_recent_embs = user_recent_embs
        self.ranker_mode, self.ranker_beta = _load_ranker_blend()
        self._rp3 = _build_rp3(config, train_df, user2idx, item2idx)
        
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
        history = self.user_history.get(user_id, set())
        tower = unseen_candidates(
            u_emb, self.faiss_index, self.idx2item, history, self._retrieval_k()
        )
        source_columns = None
        if self._rp3 is None:
            if not tower:
                return self.recommend_cold_start(top_k)
            candidates = tower
            ranks = list(range(len(candidates)))
        else:
            candidates, ranks, source_columns = self._with_rp3(u_idx, u_emb, tower, history)
            if not candidates:
                return self.recommend_cold_start(top_k)

        pairs_df = pd.DataFrame({
            'user_id': [user_id] * len(candidates),
            'item_id': candidates,
            'retrieval_rank': ranks,
        })
        if source_columns:
            for column, values in source_columns.items():
                pairs_df[column] = values
        X = build_pair_features(
            self.user_features, self.item_features,
            self.user_embs, self.item_embs,
            self.user2idx, self.item2idx, pairs_df, self.user_genre_profiles,
            user_recent_embs=self.user_recent_embs,
        )
        model_frame, baseline = ranker_inputs(X)
        if self.ranker_mode == 'retrieval':
            pairs_df['score'] = baseline
        else:
            scores = self.catboost_ranker.predict(model_frame)
            pairs_df['score'] = baseline + self.ranker_beta * scores
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

    def _with_rp3(self, user_index: int, user_emb: np.ndarray, tower: list[int], history: set):
        """Union the served FAISS list with RP3. Source columns follow the loaded model."""
        rp3_scores = self._rp3.score_row(user_index, history)
        rp3_ids = self._rp3.top_ids(rp3_scores)
        candidates, ranks = union_with_rp3(tower, rp3_ids, self._retrieval_k())
        wanted = trained_source_columns(self.catboost_ranker)
        if not wanted:
            return candidates, ranks, None
        tt_scores = two_tower_score_row(
            user_emb, self.faiss_index, len(self.item2idx), history, self.item2idx,
        )
        columns = source_feature_columns(
            candidates, tower, rp3_ids, tt_scores, rp3_scores, self._rp3.item2idx,
        )
        return candidates, ranks, {column: columns[column] for column in wanted}

    def _retrieval_k(self) -> int:
        if 'retrieval_top_k' in self.config:
            return int(self.config['retrieval_top_k'])
        pipeline_cfg = self.config.get('pipeline') or {}
        return int(pipeline_cfg.get('retrieval_top_k', 200))

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
