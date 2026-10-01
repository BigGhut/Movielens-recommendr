import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from src.data.preprocessing import create_id_mappings

def build_hetero_graph(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame) -> tuple[HeteroData, dict, dict, dict]:
    """
    Строит гетерогенный граф (HeteroData) из датафреймов MovieLens.
    Узлы: user, movie, genre.
    Ребра: 
      - (user, rates, movie)
      - (movie, has, genre)
    """
    # 1. Маппинги для User и Movie (чтобы индексы были от 0 до N-1)
    user2idx, item2idx = create_id_mappings(ratings_df)
    
    # Оставляем только те сущности, которые есть в ratings_df
    valid_users = users_df[users_df['user_id'].isin(user2idx.keys())].copy()
    valid_movies = movies_df[movies_df['item_id'].isin(item2idx.keys())].copy()
    
    # 2. Маппинг для Genre
    all_genres = set()
    for genres_str in valid_movies['genres']:
        if pd.isna(genres_str):
            continue
        # В MovieLens жанры разделены '|'
        parts = genres_str.split('|')
        for p in parts:
            all_genres.add(p)
            
    genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
    
    data = HeteroData()
    
    # ================= УЗЛЫ =================
    
    # --- User ---
    num_users = len(user2idx)
    # Фичи: gender, age, occupation
    # Для простоты пока используем просто ID как фичу (будет пропущен через Embedding слой)
    data['user'].node_id = torch.arange(num_users)
    
    # Дополнительно можно добавить признаки, если захотим конкатенировать:
    # Замапим пользователей по их индексам
    valid_users['u_idx'] = valid_users['user_id'].map(user2idx)
    valid_users = valid_users.sort_values('u_idx')
    
    # Пример: закодируем gender
    # F -> 0, M -> 1
    gender_map = {'F': 0, 'M': 1}
    gender_tensor = torch.tensor([gender_map.get(g, 2) for g in valid_users['gender'].values], dtype=torch.long)
    data['user'].gender = gender_tensor
    
    # --- Movie ---
    num_movies = len(item2idx)
    data['movie'].node_id = torch.arange(num_movies)
    
    # Пытаемся загрузить мультимодальные эмбеддинги, если они есть
    from pathlib import Path
    text_path = Path("artifacts/models/text_embeddings.pt")
    vision_path = Path("artifacts/models/vision_embeddings.pt")
    
    has_multimodal = False
    if text_path.exists() and vision_path.exists():
        text_dict = torch.load(text_path, weights_only=False)
        vision_dict = torch.load(vision_path, weights_only=False)
        
        # Размерность = 768 (Text) + 768 (Vision) = 1536
        text_dim = list(text_dict.values())[0].shape[-1]
        vision_dim = list(vision_dict.values())[0].shape[-1]
        
        multimodal_x = torch.zeros((num_movies, text_dim + vision_dim), dtype=torch.float)
        
        for m_id, m_idx in item2idx.items():
            t_emb = text_dict.get(m_id, np.zeros(text_dim))
            v_emb = vision_dict.get(m_id, np.zeros(vision_dim))
            multimodal_x[m_idx] = torch.tensor(np.concatenate([t_emb, v_emb]), dtype=torch.float)
            
        data['movie'].multimodal_x = multimodal_x
        has_multimodal = True
    
    # --- Genre ---
    num_genres = len(genre2idx)
    data['genre'].node_id = torch.arange(num_genres)
    
    # ================= РЕБРА =================
    
    # --- (user, rates, movie) ---
    # Извлечем edges
    u_indices = ratings_df['user_id'].map(user2idx).values
    i_indices = ratings_df['item_id'].map(item2idx).values
    
    # PyG ожидает edge_index размера [2, num_edges]
    edge_index_user_movie = torch.tensor(np.vstack((u_indices, i_indices)), dtype=torch.long)
    data['user', 'rates', 'movie'].edge_index = edge_index_user_movie
    
    # Атрибуты ребер: рейтинг и время
    ratings_tensor = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
    # Нормализация времени (min-max)
    timestamps = ratings_df['timestamp'].values
    min_t, max_t = timestamps.min(), timestamps.max()
    time_norm = (timestamps - min_t) / (max_t - min_t + 1e-9)
    time_tensor = torch.tensor(time_norm, dtype=torch.float)
    
    # Edge features: [rating, timestamp]
    edge_attr = torch.stack([ratings_tensor, time_tensor], dim=1)
    data['user', 'rates', 'movie'].edge_attr = edge_attr
    
    # Добавим обратные ребра (movie, rated_by, user) для двунаправленного Message Passing
    data['movie', 'rated_by', 'user'].edge_index = edge_index_user_movie.flip([0])
    data['movie', 'rated_by', 'user'].edge_attr = edge_attr
    
    # --- (movie, has, genre) ---
    movie_genre_u = []
    movie_genre_v = []
    
    for _, row in valid_movies.iterrows():
        m_id = row['item_id']
        if m_id not in item2idx:
            continue
        m_idx = item2idx[m_id]
        
        genres_str = row['genres']
        if pd.isna(genres_str):
            continue
            
        parts = genres_str.split('|')
        for p in parts:
            if p in genre2idx:
                g_idx = genre2idx[p]
                movie_genre_u.append(m_idx)
                movie_genre_v.append(g_idx)
                
    edge_index_movie_genre = torch.tensor([movie_genre_u, movie_genre_v], dtype=torch.long)
    data['movie', 'has', 'genre'].edge_index = edge_index_movie_genre
    
    # Обратные ребра
    data['genre', 'belongs_to', 'movie'].edge_index = edge_index_movie_genre.flip([0])
    
    return data, user2idx, item2idx, genre2idx
