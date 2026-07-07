import pytest
import numpy as np
from pathlib import Path
from src.retrieval.index import FAISSIndex

@pytest.fixture
def index():
    return FAISSIndex(embedding_dim=16)

@pytest.fixture
def dummy_embeddings():
    np.random.seed(42)
    # 10 векторов размерности 16
    embs = np.random.rand(10, 16).astype(np.float32)
    # L2 normalize (faiss flat IP ожидает нормализованные для косинусного сходства)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms

def test_build_and_search(index, dummy_embeddings):
    index.build(dummy_embeddings)
    
    # Ищем самого себя для вектора 0
    query = dummy_embeddings[0:1]
    distances, indices = index.search(query, top_k=3)
    
    assert indices.shape == (1, 3)
    assert indices[0][0] == 0 # Самый близкий должен быть он сам
    assert pytest.approx(distances[0][0], 0.01) == 1.0 # Косинусное сходство с самим собой 1.0

def test_save_load(index, dummy_embeddings, tmp_path):
    index.build(dummy_embeddings)
    
    file_path = tmp_path / "test_index.index"
    index.save(file_path)
    
    loaded_index = FAISSIndex.load(file_path)
    
    assert len(loaded_index) == 10
    
    query = dummy_embeddings[0:1]
    d1, i1 = index.search(query, top_k=2)
    d2, i2 = loaded_index.search(query, top_k=2)
    
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_almost_equal(d1, d2)

def test_index_size(index, dummy_embeddings):
    assert len(index) == 0
    index.build(dummy_embeddings)
    assert len(index) == 10
