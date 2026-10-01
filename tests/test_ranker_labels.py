import numpy as np
import pandas as pd

from src.models.ranker import RANK_BASELINE_SCALE, finalize_ranker_frame, ranker_inputs
from src.models.train_ranker import build_labeled_pairs
from src.retrieval.index import FAISSIndex


def _index():
    vectors = np.array([
        [0.6, 0.8, 0.0, 0.0],
        [0.8, 0.6, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = FAISSIndex(embedding_dim=4)
    index.build(vectors)
    return index


def test_labels_use_the_future_item_and_drop_history():
    user_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    user2idx = {7: 0}
    idx2item = {0: 10, 1: 11, 2: 12, 3: 13}
    history = pd.DataFrame({"user_id": [7], "item_id": [12]})
    labels = pd.DataFrame({"user_id": [7], "item_id": [11], "rating": [5]})

    pairs, relevance = build_labeled_pairs(
        history, labels, user_embs, user2idx, idx2item, _index(),
        top_k=2, positive_threshold=4,
    )

    assert pairs["item_id"].tolist() == [11, 10]
    assert 12 not in pairs["item_id"].tolist()
    assert pairs["retrieval_rank"].tolist() == [0.0, 1.0]
    assert relevance.tolist() == [1.0, 0.0]


def test_users_without_a_retrieved_positive_are_skipped():
    user_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    user2idx = {7: 0}
    idx2item = {0: 10, 1: 11, 2: 12, 3: 13}
    history = pd.DataFrame({"user_id": [7], "item_id": [12]})
    labels = pd.DataFrame({
        "user_id": [7, 7],
        "item_id": [11, 13],
        "rating": [2, 5],
    })

    pairs, relevance = build_labeled_pairs(
        history, labels, user_embs, user2idx, idx2item, _index(),
        top_k=1, positive_threshold=4,
    )

    assert pairs.empty
    assert relevance.size == 0


def test_ranker_frame_drops_the_label():
    frame = pd.DataFrame({
        "user_id": [1],
        "item_id": [2],
        "title": ["Movie"],
        "label": [1.0],
        "cosine_similarity": [0.2],
        "retrieval_rank": [3.0],
    })
    prepared = finalize_ranker_frame(frame)
    assert "label" not in prepared.columns
    assert "cosine_similarity" in prepared.columns
    assert prepared["retrieval_rank"].tolist() == [3.0]


def test_ranker_inputs_use_faiss_order_as_baseline():
    frame = pd.DataFrame({
        "user_id": [1, 1],
        "retrieval_rank": [0.0, 4.0],
        "label": [0.0, 1.0],
        "genre_affinity": [0.1, 0.8],
    })
    features, baseline = ranker_inputs(frame)
    assert "retrieval_rank" not in features.columns
    assert "label" not in features.columns
    assert baseline.tolist() == [0.0, -4.0 / RANK_BASELINE_SCALE]
    assert features["genre_affinity"].tolist() == [0.1, 0.8]
