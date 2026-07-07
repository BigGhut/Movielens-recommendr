import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.models.two_tower import TwoTowerModel
import random
import os

class PairDataset(Dataset):
    def __init__(self, user_ids, item_ids):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        
    def __len__(self):
        return len(self.user_ids)
        
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_two_tower(config: dict, train_df, val_df, user2idx: dict, item2idx: dict, save_dir: Path) -> TwoTowerModel:
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на устройстве: {device}")
    
    # Подготовка данных (используем только те взаимодействия, которые есть в train и где id известны)
    train_valid = train_df[train_df['user_id'].isin(user2idx) & train_df['item_id'].isin(item2idx)]
    train_u = [user2idx[u] for u in train_valid['user_id']]
    train_i = [item2idx[i] for i in train_valid['item_id']]
    
    train_dataset = PairDataset(train_u, train_i)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Модель
    model = TwoTowerModel(
        num_users=len(user2idx),
        num_items=len(item2idx),
        emb_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0.0
        for batch_u, batch_i in train_loader:
            batch_u, batch_i = batch_u.to(device), batch_i.to(device)
            
            optimizer.zero_grad()
            user_embs, item_embs = model(batch_u, batch_i)
            loss = model.compute_loss(user_embs, item_embs, config['temperature'])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")
        
    # Сохранение модели
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "two_tower.pth")
    print(f"Модель сохранена в {save_dir / 'two_tower.pth'}")
    return model

if __name__ == '__main__':
    from src.data.preprocessing import load_ratings, temporal_split, create_id_mappings
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)["two_tower"]
        
    ratings = load_ratings(Path("data/raw/ml-1m"))
    train_df, val_df, test_df = temporal_split(ratings)
    user2idx, item2idx = create_id_mappings(train_df)
    
    model = train_two_tower(config, train_df, val_df, user2idx, item2idx, Path("artifacts/models"))

    from src.retrieval.embeddings import generate_all_user_embeddings, generate_all_item_embeddings
    from src.retrieval.index import FAISSIndex
    import numpy as np

    print("Генерация эмбеддингов...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_embs = generate_all_user_embeddings(model, user2idx, device)
    item_embs = generate_all_item_embeddings(model, item2idx, device)
    
    np.save("artifacts/models/user_embeddings.npy", user_embs)
    np.save("artifacts/models/item_embeddings.npy", item_embs)
    
    print("Построение FAISS индекса...")
    faiss_index = FAISSIndex(embedding_dim=config['output_dim'])
    faiss_index.build(item_embs)
    
    index_path = Path("artifacts/indexes/faiss_index.index")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_index.save(index_path)
    print(f"Индекс сохранен в {index_path}")
