import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users: int, emb_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_users, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_ids)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1) # L2 normalization

class ItemTower(nn.Module):
    def __init__(self, num_items: int, emb_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(item_ids)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1) # L2 normalization

class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim, hidden_dim, output_dim)
        self.item_tower = ItemTower(num_items, emb_dim, hidden_dim, output_dim)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_embs = self.user_tower(user_ids)
        item_embs = self.item_tower(item_ids)
        return user_embs, item_embs
        
    def compute_loss(self, user_embs: torch.Tensor, item_embs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Вычисляет InfoNCE loss с in-batch negatives.
        Матрица схожести user_embs @ item_embs.T
        Правильные пары - на диагонали.
        """
        # similarity matrix: [batch_size, batch_size]
        logits = torch.matmul(user_embs, item_embs.T) / temperature
        
        # метки классов - это индексы диагонали: 0, 1, ..., batch_size-1
        batch_size = user_embs.size(0)
        labels = torch.arange(batch_size, device=user_embs.device)
        
        # cross_entropy: максимизирует схожесть позитивных пар и минимизирует для негативных
        loss = F.cross_entropy(logits, labels)
        return loss
