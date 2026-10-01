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

def test_genre_buffer_changes_item_embedding(model):
    item_ids = torch.tensor([1, 2])
    before = model.encode_items(item_ids).detach().clone()
    model.item_genre[1] = 1.0
    model.item_year[1] = 1.5
    after = model.encode_items(item_ids)
    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1], after[1])


def test_recent_history_changes_user_embedding(model):
    user_ids = torch.tensor([1])
    bare = model.user_tower(user_ids)
    recent_items = torch.tensor([[3, 0]])
    recent_mask = torch.tensor([[1.0, 0.0]])
    recent_genre = torch.zeros(1, model.item_genre.shape[1])
    recent_year = torch.zeros(1)
    with_history = model.encode_users(user_ids, recent_genre, recent_year, recent_items, recent_mask)
    assert with_history.shape == bare.shape
    assert not torch.allclose(bare, with_history)


def test_hard_negatives_change_loss(model):
    user_ids = torch.tensor([1, 2, 3])
    item_ids = torch.tensor([4, 5, 6])
    user_embs, item_embs = model(user_ids, item_ids)
    base = model.compute_loss(user_embs, item_embs)
    hard = torch.nn.functional.normalize(torch.randn(3, 4, item_embs.shape[1]), dim=-1)
    with_hard = model.compute_loss(user_embs, item_embs, hard_item_embs=hard)
    assert torch.isfinite(with_hard)
    assert not torch.allclose(base, with_hard)


def test_different_users_different_embeddings(model):
    user_ids = torch.tensor([1, 2])
    u_embs = model.user_tower(user_ids)
    
    # Вероятность того, что случайно инициализированные эмбеддинги будут одинаковыми крайне мала
    assert not torch.allclose(u_embs[0], u_embs[1])
