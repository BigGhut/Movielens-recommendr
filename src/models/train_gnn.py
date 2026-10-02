from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.graph_builder import build_hetero_graph
from src.data.preprocessing import load_movies, load_ratings, load_users, temporal_split
from src.models.hetero_gnn import HeteroLinkPredictionModel


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    
    data_dir = Path("data/raw/ml-1m")
    ratings = load_ratings(data_dir)
    movies_df = load_movies(data_dir)
    users_df = load_users(data_dir)
    
    train_df, _val_df, _test_df = temporal_split(ratings)
    
    print("Построение графа...")
    data, user2idx, item2idx, genre2idx = build_hetero_graph(train_df, movies_df, users_df)
    data = data.to(device)
    
    num_users = len(user2idx)
    num_movies = len(item2idx)
    num_genres = len(genre2idx)
    
    multimodal_x = getattr(data['movie'], 'multimodal_x', None)
    if multimodal_x is not None:
        multimodal_x = multimodal_x.to(device)
        multimodal_dim = multimodal_x.size(1)
        print(f"Используем мультимодальные признаки размерности {multimodal_dim}")
    else:
        multimodal_dim = None

    model = HeteroLinkPredictionModel(
        hidden_channels=64, 
        num_users=num_users, 
        num_movies=num_movies, 
        num_genres=num_genres,
        multimodal_dim=multimodal_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Позитивные ребра для обучения
    pos_edge_index = data['user', 'rates', 'movie'].edge_index
    num_pos_edges = pos_edge_index.size(1)
    
    epochs = 50
    batch_size = 8192
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Перемешиваем индексы ребер
        perm = torch.randperm(num_pos_edges)
        
        total_loss = 0
        num_batches = 0
        
        x_dict = {
            'user': data['user'].node_id,
            'movie': data['movie'].node_id,
            'genre': data['genre'].node_id
        }
        
        for i in range(0, num_pos_edges, batch_size):
            optimizer.zero_grad()
            
            # Forward pass per batch
            z_dict = model.gnn(x_dict, data.edge_index_dict, multimodal_x=multimodal_x)
            user_z = z_dict['user']
            movie_z = z_dict['movie']
            
            # Батч позитивных ребер
            batch_perm = perm[i:i+batch_size]
            batch_pos_edges = pos_edge_index[:, batch_perm]
            
            # Генерируем случайные негативные ребра
            neg_movie_indices = torch.randint(0, num_movies, (batch_pos_edges.size(1),), device=device)
            batch_neg_edges = torch.stack([batch_pos_edges[0], neg_movie_indices], dim=0)
            
            # Считаем скоры для позитивных
            pos_user = user_z[batch_pos_edges[0]]
            pos_movie = movie_z[batch_pos_edges[1]]
            pos_scores = (pos_user * pos_movie).sum(dim=-1)
            
            # Считаем скоры для негативных
            neg_user = user_z[batch_neg_edges[0]]
            neg_movie = movie_z[batch_neg_edges[1]]
            neg_scores = (neg_user * neg_movie).sum(dim=-1)
            
            # BPR Loss
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.detach())
            num_batches += 1
            
        print(f"Epoch {epoch:03d}, Loss: {total_loss / num_batches:.4f}")
            
    # Сохраняем финальные эмбеддинги
    print("Генерация финальных эмбеддингов...")
    model.eval()
    with torch.no_grad():
        x_dict = {
            'user': data['user'].node_id,
            'movie': data['movie'].node_id,
            'genre': data['genre'].node_id
        }
        z_dict = model.gnn(x_dict, data.edge_index_dict, multimodal_x=multimodal_x)
        user_embeddings = z_dict['user'].cpu().numpy()
        item_embeddings = z_dict['movie'].cpu().numpy()
        
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    np.save("artifacts/models/gnn_user_embeddings.npy", user_embeddings)
    np.save("artifacts/models/gnn_item_embeddings.npy", item_embeddings)
    print("Эмбеддинги GNN сохранены отдельно от two-tower: artifacts/models/gnn_*.npy")

    from src.retrieval.index import FAISSIndex
    print("Построение FAISS индекса GNN...")
    faiss_index = FAISSIndex(embedding_dim=item_embeddings.shape[1])
    faiss_index.build(item_embeddings)
    Path("artifacts/indexes").mkdir(parents=True, exist_ok=True)
    faiss_index.save(Path("artifacts/indexes/gnn_faiss.index"))
    print("GNN индекс сохранен в artifacts/indexes/gnn_faiss.index")

if __name__ == '__main__':
    train()
