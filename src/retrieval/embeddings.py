import torch
import numpy as np
from src.models.two_tower import TwoTowerModel

@torch.no_grad()
def generate_all_item_embeddings(model: TwoTowerModel, item2idx: dict, device: torch.device) -> np.ndarray:
    """Генерирует эмбеддинги для всех фильмов."""
    model.eval()
    model.to(device)
    num_items = len(item2idx)
    all_item_ids = torch.arange(num_items, device=device)
    
    # Можно батчевать, если элементов очень много, но для ML-1M (3700 фильмов) влезет целиком
    item_embs = model.item_tower(all_item_ids)
    return item_embs.cpu().numpy()

@torch.no_grad()
def generate_user_embedding(model: TwoTowerModel, user_idx: int, device: torch.device) -> np.ndarray:
    """Генерирует эмбеддинг для одного пользователя."""
    model.eval()
    model.to(device)
    user_id_tensor = torch.tensor([user_idx], device=device, dtype=torch.long)
    user_emb = model.user_tower(user_id_tensor)
    return user_emb.cpu().numpy()

@torch.no_grad()
def generate_all_user_embeddings(model: TwoTowerModel, user2idx: dict, device: torch.device) -> np.ndarray:
    """Генерирует эмбеддинги для всех пользователей."""
    model.eval()
    model.to(device)
    num_users = len(user2idx)
    all_user_ids = torch.arange(num_users, device=device)
    user_embs = model.user_tower(all_user_ids)
    return user_embs.cpu().numpy()
