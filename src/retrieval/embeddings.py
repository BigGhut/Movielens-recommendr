import numpy as np
import torch

from src.models.two_tower import TwoTowerModel


@torch.no_grad()
def generate_all_item_embeddings(model: TwoTowerModel, item2idx: dict, device: torch.device) -> np.ndarray:
    """L2-normalized item vectors, including genre and year stored on the model."""
    model.eval()
    model.to(device)
    item_ids = torch.arange(len(item2idx), device=device)
    return model.encode_items(item_ids).cpu().numpy()

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
    user_embs = model.encode_users(
        all_user_ids,
        torch.zeros(num_users, model.item_genre.shape[1], device=device),
        torch.zeros(num_users, device=device),
        torch.zeros(num_users, 1, dtype=torch.long, device=device),
        torch.zeros(num_users, 1, device=device),
    ) if hasattr(model, "encode_users") else model.user_tower(all_user_ids)
    return user_embs.cpu().numpy()
