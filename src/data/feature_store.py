import pandas as pd
import numpy as np

def build_user_features(train_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    user_stats = train_df.groupby('user_id').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        activity_days_span=('timestamp', lambda x: (x.max() - x.min()) / (24 * 3600))
    ).reset_index()

    merged = train_df.merge(movies_df, on='item_id')
    most_common_genre = merged.groupby('user_id')['genres'].agg(
        lambda x: x.value_counts().index[0] if not x.empty else "Unknown"
    ).reset_index(name='most_common_genre')
    
    item_pops = train_df.groupby('item_id').size()
    item_pops_dict = item_pops.to_dict()
    train_with_pop = train_df.copy()
    train_with_pop['item_pop'] = train_with_pop['item_id'].map(item_pops_dict)
    user_mainstream = train_with_pop.groupby('user_id')['item_pop'].mean().reset_index(name='user_mainstream_score')
    
    user_feats = user_stats.merge(most_common_genre, on='user_id')
    user_feats = user_feats.merge(user_mainstream, on='user_id')
    user_feats = user_feats.merge(users_df, on='user_id')
    
    return user_feats

def build_item_features(train_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    item_stats = train_df.groupby('item_id').agg(
        item_num_ratings=('rating', 'count'),
        item_avg_rating=('rating', 'mean'),
        item_positive_ratio=('rating', lambda x: (x >= 4).mean())
    ).reset_index()
    
    item_stats['item_popularity_percentile'] = item_stats['item_num_ratings'].rank(pct=True)
    
    movies_copy = movies_df.copy()
    movies_copy['release_year'] = movies_copy['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_copy['release_year'] = movies_copy['release_year'].fillna(1990)
    
    item_feats = movies_copy.merge(item_stats, on='item_id', how='left')
    item_feats['item_num_ratings'] = item_feats['item_num_ratings'].fillna(0)
    item_feats['item_avg_rating'] = item_feats['item_avg_rating'].fillna(item_feats['item_avg_rating'].mean())
    item_feats['item_positive_ratio'] = item_feats['item_positive_ratio'].fillna(item_feats['item_positive_ratio'].mean())
    item_feats['item_popularity_percentile'] = item_feats['item_popularity_percentile'].fillna(0.0)
    
    return item_feats

def build_user_genre_profiles(train_df: pd.DataFrame, movies_df: pd.DataFrame) -> dict:
    merged = train_df.merge(movies_df, on='item_id')
    profiles = {}
    
    # Группируем по юзеру
    for user_id, group in merged.groupby('user_id'):
        genre_counts = {}
        for genres_str in group['genres']:
            for g in str(genres_str).split('|'):
                genre_counts[g] = genre_counts.get(g, 0) + 1
        
        # Нормализуем, чтобы получить доли
        total = sum(genre_counts.values())
        if total > 0:
            for g in genre_counts:
                genre_counts[g] /= total
                
        profiles[user_id] = genre_counts
    return profiles

def build_recent_centroids(history_df: pd.DataFrame, item_embs: np.ndarray,
                           user2idx: dict, item2idx: dict, recent_k: int = 10) -> np.ndarray:
    """L2-normalized mean of each user's last `recent_k` train item vectors."""
    centroids = np.zeros((len(user2idx), item_embs.shape[1]), dtype=np.float32)
    ordered = history_df.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id", sort=False):
        user_key = int(user_id)
        if user_key not in user2idx:
            continue
        taken = [item2idx[int(item_id)] for item_id in group["item_id"] if int(item_id) in item2idx][-recent_k:]
        if not taken:
            continue
        mean = item_embs[taken].mean(axis=0)
        centroids[user2idx[user_key]] = mean / (np.linalg.norm(mean) + 1e-8)
    return centroids


def build_pair_features(user_feats: pd.DataFrame, item_feats: pd.DataFrame, 
                        user_embs: np.ndarray, item_embs: np.ndarray,
                        user_idx_map: dict, item_idx_map: dict, 
                        pairs_df: pd.DataFrame, user_genre_profiles: dict = None,
                        user_recent_embs: np.ndarray = None) -> pd.DataFrame:
    df = pairs_df.copy()
    df = df.merge(user_feats, on='user_id', how='left')
    df = df.merge(item_feats, on='item_id', how='left')
    
    df['rating_diff'] = df['item_avg_rating'] - df['avg_rating']
    df['cosine_similarity'] = _cosine_similarity(df, user_embs, item_embs, user_idx_map, item_idx_map)
    df['genre_affinity'] = _genre_affinity(df, user_genre_profiles)
    df['recent_neighbor_score'] = _cosine_similarity(df, user_recent_embs, item_embs, user_idx_map, item_idx_map) if user_recent_embs is not None else 0.0
    return df


def _cosine_similarity(df: pd.DataFrame, user_embs: np.ndarray, item_embs: np.ndarray,
                       user_idx_map: dict, item_idx_map: dict) -> np.ndarray:
    user_idx = df['user_id'].map(user_idx_map)
    item_idx = df['item_id'].map(item_idx_map)
    valid = user_idx.notna() & item_idx.notna()
    sims = np.zeros(len(df), dtype=np.float64)
    if not valid.any():
        return sims
    users = user_embs[user_idx[valid].to_numpy(dtype=np.int64)]
    items = item_embs[item_idx[valid].to_numpy(dtype=np.int64)]
    denom = np.linalg.norm(users, axis=1) * np.linalg.norm(items, axis=1) + 1e-8
    sims[valid.to_numpy()] = np.sum(users * items, axis=1) / denom
    return sims


def _genre_affinity(df: pd.DataFrame, user_genre_profiles: dict | None) -> np.ndarray:
    affinities = np.zeros(len(df), dtype=np.float64)
    if not user_genre_profiles or 'genres' not in df.columns:
        return affinities
    profiles = {int(user_id): weights for user_id, weights in user_genre_profiles.items()}
    for i, (user_id, genre_str) in enumerate(zip(df['user_id'].tolist(), df['genres'].tolist())):
        profile = profiles.get(int(user_id))
        if not profile or genre_str is None or (isinstance(genre_str, float) and np.isnan(genre_str)):
            continue
        affinities[i] = sum(profile.get(part, 0.0) for part in str(genre_str).split('|'))
    return affinities

def get_feature_names() -> list[str]:
    return [
        "num_ratings",
        "avg_rating",
        "activity_days_span",
        "most_common_genre",
        "user_mainstream_score",
        "gender",
        "age",
        "occupation",
        "zip",
        "item_num_ratings",
        "item_avg_rating",
        "item_positive_ratio",
        "item_popularity_percentile",
        "genres",
        "release_year",
        "rating_diff",
        "cosine_similarity",
        "genre_affinity",
        "recent_neighbor_score",
    ]
