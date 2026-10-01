import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.catalog import (
    build_item_catalog,
    build_pair_contexts,
    build_user_contexts,
    popularity_weights,
)
from src.models.two_tower import TwoTowerModel


class PairDataset(Dataset):
    def __init__(self, contexts):
        self.user = torch.from_numpy(contexts.user_idx)
        self.item = torch.from_numpy(contexts.item_idx)
        self.recent_genre = torch.from_numpy(contexts.recent_genre)
        self.recent_year = torch.from_numpy(contexts.recent_year)
        self.recent_items = torch.from_numpy(contexts.recent_items)
        self.recent_mask = torch.from_numpy(contexts.recent_mask)

    def __len__(self):
        return int(self.user.shape[0])

    def __getitem__(self, idx):
        return (
            self.user[idx],
            self.item[idx],
            self.recent_genre[idx],
            self.recent_year[idx],
            self.recent_items[idx],
            self.recent_mask[idx],
        )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sample_hard_negatives(item_ids: torch.Tensor, weights: torch.Tensor, num_hard: int) -> torch.Tensor:
    num_items = int(weights.shape[0])
    hard_ids = torch.multinomial(weights, item_ids.shape[0] * num_hard, replacement=True)
    hard_ids = hard_ids.view(item_ids.shape[0], num_hard)
    collision = hard_ids == item_ids.unsqueeze(1)
    if collision.any() and num_items > 1:
        hard_ids = torch.where(collision, (hard_ids + 1) % num_items, hard_ids)
    return hard_ids


def train_two_tower(config: dict, train_df, movies_df, user2idx: dict, item2idx: dict, save_dir: Path) -> TwoTowerModel:
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на устройстве: {device}")

    catalog = build_item_catalog(movies_df, item2idx, train_df)
    recent_k = int(config.get("recent_k", 10))
    num_hard = int(config.get("hard_negatives", 8))
    contexts = build_pair_contexts(train_df, user2idx, item2idx, catalog, recent_k)
    print(f"Жанров: {len(catalog.genres)}, пар: {len(contexts.user_idx)}, recent_k={recent_k}, hard_negatives={num_hard}")

    dataset = PairDataset(contexts)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=len(dataset) > config["batch_size"],
        num_workers=0,
    )
    model = TwoTowerModel(
        num_users=len(user2idx),
        num_items=len(item2idx),
        emb_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_genres=len(catalog.genres),
    ).to(device)
    model.item_genre.copy_(torch.as_tensor(catalog.item_genre, device=device))
    model.item_year.copy_(torch.as_tensor(catalog.item_year, device=device))
    popularity = torch.as_tensor(popularity_weights(train_df, item2idx), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        for batch in loader:
            batch = tuple(tensor.to(device) for tensor in batch)
            user_ids, item_ids, recent_genre, recent_year, recent_items, recent_mask = batch
            optimizer.zero_grad()
            user_embs = model.encode_users(user_ids, recent_genre, recent_year, recent_items, recent_mask)
            item_embs = model.encode_items(item_ids)
            hard_ids = _sample_hard_negatives(item_ids, popularity, num_hard)
            hard_embs = model.encode_items(hard_ids.reshape(-1)).view(item_ids.shape[0], num_hard, -1)
            loss = model.compute_loss(user_embs, item_embs, config["temperature"], hard_embs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {total_loss / len(loader):.4f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "two_tower.pth")
    print(f"Модель сохранена в {save_dir / 'two_tower.pth'}")
    model.catalog = catalog
    model.recent_k = recent_k
    return model


def export_normalized_index(model: TwoTowerModel, train_df, user2idx: dict, item2idx: dict, config: dict) -> None:
    """Write L2-normalized two-tower vectors and the FAISS index the service searches."""
    from src.retrieval.index import FAISSIndex

    device = next(model.parameters()).device
    model.eval()
    recent_k = int(getattr(model, "recent_k", config.get("recent_k", 10)))
    catalog = model.catalog
    users = build_user_contexts(train_df, user2idx, item2idx, catalog, recent_k)
    with torch.no_grad():
        item_ids = torch.arange(len(item2idx), device=device)
        item_embs = model.encode_items(item_ids).cpu().numpy()
        user_ids = torch.arange(len(user2idx), device=device)
        user_embs = model.encode_users(
            user_ids,
            torch.as_tensor(users.recent_genre, device=device),
            torch.as_tensor(users.recent_year, device=device),
            torch.as_tensor(users.recent_items, device=device),
            torch.as_tensor(users.recent_mask, device=device),
        ).cpu().numpy()

    item_norms = np.linalg.norm(item_embs, axis=1)
    user_norms = np.linalg.norm(user_embs, axis=1)
    print(f"Нормы эмбеддингов: item {item_norms.mean():.4f}, user {user_norms.mean():.4f}")
    np.save("artifacts/models/item_embeddings.npy", item_embs)
    np.save("artifacts/models/user_embeddings.npy", user_embs)
    index = FAISSIndex(embedding_dim=item_embs.shape[1])
    index.build(item_embs)
    index_path = Path("artifacts/indexes/faiss_index.index")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    print(f"Индекс сохранен в {index_path}")
    return user_embs, item_embs


def report_test_recall(train_df, test_df, user2idx, item2idx, user_embs) -> None:
    from src.retrieval.candidates import unseen_candidates
    from src.retrieval.index import FAISSIndex

    index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
    idx2item = {value: key for key, value in item2idx.items()}
    history = {int(user_id): set(map(int, item_ids)) for user_id, item_ids in train_df.groupby("user_id")["item_id"]}
    positives = test_df[test_df["rating"] >= 4]
    hits = {10: 0, 200: 0}
    counted = 0
    for row in positives.itertuples(index=False):
        user_id, item_id = int(row.user_id), int(row.item_id)
        if user_id not in user2idx or item_id not in item2idx:
            continue
        counted += 1
        query = user_embs[user2idx[user_id] : user2idx[user_id] + 1]
        candidates = unseen_candidates(query, index, idx2item, history.get(user_id, set()), 200)
        if item_id in candidates:
            rank = candidates.index(item_id)
            if rank < 10:
                hits[10] += 1
            hits[200] += 1
    if counted == 0:
        print("Test recall: нет позитивов")
        return
    print(
        f"Test recall@10 {hits[10] / counted:.4f} ({hits[10]}/{counted}), "
        f"recall@200 {hits[200] / counted:.4f} ({hits[200]}/{counted})"
    )


if __name__ == "__main__":
    from src.data.preprocessing import create_id_mappings, load_movies, load_ratings, temporal_split

    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["two_tower"]

    ratings = load_ratings(Path("data/raw/ml-1m"))
    movies = load_movies(Path("data/raw/ml-1m"))
    train_df, _val_df, _test_df = temporal_split(ratings)
    user2idx, item2idx = create_id_mappings(train_df)
    model = train_two_tower(config, train_df, movies, user2idx, item2idx, Path("artifacts/models"))
    print("Генерация нормированных эмбеддингов...")
    user_embs, _item_embs = export_normalized_index(model, train_df, user2idx, item2idx, config)
    report_test_recall(train_df, _test_df, user2idx, item2idx, user_embs)
