import torch
import pytest
from src.models.two_tower import TwoTowerModel

@pytest.fixture
def model():
    return TwoTowerModel(num_users=10, num_items=20, emb_dim=16, hidden_dim=32, output_dim=16)

def test_model_output_shape(model):
    user_ids = torch.tensor([1, 2, 3])
    item_ids = torch.tensor([4, 5, 6])
    
    u_embs, i_embs = model(user_ids, item_ids)
    
    assert u_embs.shape == (3, 16)
    assert i_embs.shape == (3, 16)

def test_l2_normalized(model):
    user_ids = torch.tensor([1])
    item_ids = torch.tensor([2])
    
    u_embs, i_embs = model(user_ids, item_ids)
    
    u_norm = torch.norm(u_embs, p=2, dim=1).item()
    i_norm = torch.norm(i_embs, p=2, dim=1).item()
    
    assert pytest.approx(u_norm, 0.01) == 1.0
    assert pytest.approx(i_norm, 0.01) == 1.0

def test_infonce_loss_positive(model):
    user_ids = torch.tensor([1, 2, 3])
    item_ids = torch.tensor([4, 5, 6])
    
    u_embs, i_embs = model(user_ids, item_ids)
    loss = model.compute_loss(u_embs, i_embs)
    
    assert loss.item() > 0

def test_different_users_different_embeddings(model):
    user_ids = torch.tensor([1, 2])
    u_embs = model.user_tower(user_ids)
    
    # Вероятность того, что случайно инициализированные эмбеддинги будут одинаковыми крайне мала
    assert not torch.allclose(u_embs[0], u_embs[1])
