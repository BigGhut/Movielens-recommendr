import faiss
import numpy as np
from pathlib import Path

class FAISSIndex:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Используем Inner Product, так как эмбеддинги L2-нормализованы (IP эквивалентно косинусному сходству)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
    def build(self, embeddings: np.ndarray):
        """Строит индекс на переданных эмбеддингах."""
        assert embeddings.shape[1] == self.embedding_dim
        self.index.add(embeddings.astype(np.float32))
        
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Ищет top_k ближайших векторов.
        Возвращает: (distances, indices)
        """
        assert query.shape[1] == self.embedding_dim
        distances, indices = self.index.search(query.astype(np.float32), top_k)
        return distances, indices
        
    def save(self, path: Path):
        """Сохраняет индекс на диск."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        
    @classmethod
    def load(cls, path: Path) -> 'FAISSIndex':
        """Загружает индекс с диска."""
        index_faiss = faiss.read_index(str(path))
        d = index_faiss.d
        instance = cls(d)
        instance.index = index_faiss
        return instance
        
    def __len__(self) -> int:
        """Возвращает количество векторов в индексе."""
        return self.index.ntotal
