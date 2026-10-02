import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class RelationAwareGNN(nn.Module):
    def __init__(self, hidden_channels: int, num_users: int, num_movies: int, num_genres: int, num_layers: int = 2, multimodal_dim: int | None = None):
        super().__init__()
        
        self.multimodal_dim = multimodal_dim
        # Initial embeddings for nodes
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        
        if self.multimodal_dim is not None:
            self.movie_proj = nn.Linear(self.multimodal_dim, hidden_channels)
            self.movie_bn = nn.BatchNorm1d(hidden_channels)
        else:
            self.movie_emb = nn.Embedding(num_movies, hidden_channels)
            
        self.genre_emb = nn.Embedding(num_genres, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
                ('movie', 'rated_by', 'user'): SAGEConv((-1, -1), hidden_channels),
                ('movie', 'has', 'genre'): SAGEConv((-1, -1), hidden_channels),
                ('genre', 'belongs_to', 'movie'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            
    def forward(self, x_dict, edge_index_dict, multimodal_x=None):
        """
        x_dict: dictionary of node IDs for each node type.
        edge_index_dict: dictionary of edge indices for each edge type.
        multimodal_x: optional multimodal features for movies.
        """
        # Obtain initial node embeddings
        user_x = self.user_emb(x_dict['user'])
        genre_x = self.genre_emb(x_dict['genre'])
        
        if self.multimodal_dim is not None and multimodal_x is not None:
            movie_x = self.movie_proj(multimodal_x)
            # BatchNorm behaves differently in train/eval mode. HeteroGNN is put to train() in train_gnn.py
            movie_x = self.movie_bn(movie_x)
        else:
            movie_x = self.movie_emb(x_dict['movie'])
            
        x_dict_embs = {
            'user': user_x,
            'movie': movie_x,
            'genre': genre_x,
        }
        
        # Apply GNN layers
        for conv in self.convs:
            x_dict_embs = conv(x_dict_embs, edge_index_dict)
            x_dict_embs = {key: torch.nn.functional.relu(x) for key, x in x_dict_embs.items()}
            
        return x_dict_embs

class HeteroLinkPredictionModel(nn.Module):
    def __init__(self, hidden_channels: int, num_users: int, num_movies: int, num_genres: int, multimodal_dim: int | None = None):
        super().__init__()
        self.gnn = RelationAwareGNN(hidden_channels, num_users, num_movies, num_genres, num_layers=2, multimodal_dim=multimodal_dim)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, multimodal_x=None):
        """
        x_dict: node IDs
        edge_index_dict: message passing edges
        edge_label_index: the target edges for link prediction [2, num_edges]
        multimodal_x: optional multimodal features for movies.
        """
        # 1. Get node embeddings from GNN
        z_dict = self.gnn(x_dict, edge_index_dict, multimodal_x=multimodal_x)
        
        # 2. Extract embeddings for the nodes involved in the links we want to predict
        user_z = z_dict['user'][edge_label_index[0]]
        movie_z = z_dict['movie'][edge_label_index[1]]
        
        # 3. Compute inner product (dot product) as prediction score
        return (user_z * movie_z).sum(dim=-1)
