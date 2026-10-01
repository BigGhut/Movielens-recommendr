import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, num_genres: int, emb_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_genres = num_genres
        self.embedding = nn.Embedding(num_users, emb_dim)
        self.side_proj = nn.Linear(num_genres + 1, emb_dim)
        self.hist_proj = nn.Linear(emb_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        recent_genre: torch.Tensor | None = None,
        recent_year: torch.Tensor | None = None,
        recent_item_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        id_emb = self.embedding(user_ids)
        batch, emb_dim = id_emb.shape
        if recent_genre is None:
            recent_genre = torch.zeros(batch, self.num_genres, device=id_emb.device, dtype=id_emb.dtype)
        if recent_year is None:
            recent_year = torch.zeros(batch, device=id_emb.device, dtype=id_emb.dtype)
        if recent_item_emb is None:
            recent_item_emb = torch.zeros(batch, emb_dim, device=id_emb.device, dtype=id_emb.dtype)
        side = torch.cat([recent_genre, recent_year.unsqueeze(-1)], dim=-1)
        x = id_emb + self.side_proj(side) + self.hist_proj(recent_item_emb)
        x = self.fc2(self.relu(self.fc1(x)))
        return F.normalize(x, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_genres: int, emb_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_genres = num_genres
        self.embedding = nn.Embedding(num_items, emb_dim)
        self.genre_proj = nn.Linear(num_genres, emb_dim)
        self.year_proj = nn.Linear(1, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        genre: torch.Tensor | None = None,
        year: torch.Tensor | None = None,
    ) -> torch.Tensor:
        id_emb = self.embedding(item_ids)
        batch = id_emb.shape[0]
        if genre is None:
            genre = torch.zeros(batch, self.num_genres, device=id_emb.device, dtype=id_emb.dtype)
        if year is None:
            year = torch.zeros(batch, device=id_emb.device, dtype=id_emb.dtype)
        x = id_emb + self.genre_proj(genre) + self.year_proj(year.unsqueeze(-1))
        x = self.fc2(self.relu(self.fc1(x)))
        return F.normalize(x, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_genres: int = 18,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genres, emb_dim, hidden_dim, output_dim)
        self.item_tower = ItemTower(num_items, num_genres, emb_dim, hidden_dim, output_dim)
        self.register_buffer("item_genre", torch.zeros(num_items, num_genres))
        self.register_buffer("item_year", torch.zeros(num_items))

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids, self.item_genre[item_ids], self.item_year[item_ids])

    def encode_users(
        self,
        user_ids: torch.Tensor,
        recent_genre: torch.Tensor,
        recent_year: torch.Tensor,
        recent_item_ids: torch.Tensor,
        recent_mask: torch.Tensor,
    ) -> torch.Tensor:
        recent_emb = self.item_tower.embedding(recent_item_ids)
        mask = recent_mask.unsqueeze(-1).to(recent_emb.dtype)
        denom = recent_mask.sum(dim=1, keepdim=True).clamp(min=1.0).to(recent_emb.dtype)
        recent_mean = (recent_emb * mask).sum(dim=1) / denom
        return self.user_tower(user_ids, recent_genre, recent_year, recent_mean)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.encode_items(item_ids)

    def compute_loss(
        self,
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
        temperature: float = 0.07,
        hard_item_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE. In-batch negatives, plus optional hard negatives of shape [B, H, D]."""
        logits = torch.matmul(user_embs, item_embs.T) / temperature
        if hard_item_embs is not None:
            hard_logits = torch.einsum("bd,bhd->bh", user_embs, hard_item_embs) / temperature
            logits = torch.cat([logits, hard_logits], dim=1)
        labels = torch.arange(user_embs.size(0), device=user_embs.device)
        return F.cross_entropy(logits, labels)
