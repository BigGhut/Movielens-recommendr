import numpy as np
import pandas as pd
import pytest

from src.retrieval.graph_sources import (
    bm25_normalize,
    build_interaction_matrix,
    covisitation_weights,
    rp3_similarity,
    score_users,
    top_indices,
)


def _toy_maps():
    user2idx = {10: 0, 11: 1}
    item2idx = {100: 0, 101: 1, 102: 2}
    train = pd.DataFrame(
        {
            "user_id": [10, 10, 11, 11],
            "item_id": [100, 101, 101, 102],
            "rating": [5, 5, 5, 5],
            "timestamp": [1, 2, 3, 4],
        }
    )
    return train, user2idx, item2idx


def test_rp3_one_hop_matches_hand_calculation():
    train, user2idx, item2idx = _toy_maps()
    interactions, popularity = build_interaction_matrix(train, user2idx, item2idx)
    similarity = rp3_similarity(interactions, popularity, alpha=1.0, beta=0.0)
    expected = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.5, 0.0],
        ]
    )
    assert similarity == pytest.approx(expected)

    scaled = rp3_similarity(interactions, popularity, alpha=1.0, beta=1.0)
    assert scaled[:, 1] == pytest.approx(expected[:, 1] / 2.0)
    scores = score_users(interactions, similarity, np.array([0, 1]))
    assert scores[0, 0] == -np.inf
    assert scores[0, 1] == -np.inf
    assert scores[0, 2] == pytest.approx(0.25)
    assert scores[1, 0] == pytest.approx(0.25)


def test_matrix_ignores_items_absent_from_train_maps():
    train, user2idx, item2idx = _toy_maps()
    leaked = pd.concat(
        [
            train,
            pd.DataFrame({"user_id": [10], "item_id": [999], "rating": [5], "timestamp": [9]}),
        ],
        ignore_index=True,
    )
    interactions, popularity = build_interaction_matrix(leaked, user2idx, item2idx)
    assert interactions.nnz == 4
    assert popularity.sum() == 4


def test_covisitation_prefers_the_closer_pair_and_hides_history():
    item2idx = {100: 0, 101: 1, 102: 2}
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [100, 101, 100, 102, 100],
            "timestamp": [0, 10, 0, 1000, 0],
        }
    )
    weights = covisitation_weights(train, item2idx, tau_seconds=20.0)
    assert weights[0, 1] > weights[0, 2]
    assert weights[0, 1] == pytest.approx(weights[1, 0])
    assert weights[0, 0] == 0.0

    user2idx = {1: 0, 2: 1, 3: 2}
    interactions, popularity = build_interaction_matrix(train, user2idx, item2idx)
    similarity = bm25_normalize(weights, popularity, k1=1.2, b=0.0)
    scores = score_users(interactions, similarity, np.array([2]))
    _values, indices = top_indices(scores, k=2)
    assert indices[0, 0] == 1
    assert scores[0, 0] == -np.inf


def test_bm25_downweights_the_more_popular_destination():
    weights = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    popularity = np.array([1.0, 1.0, 100.0])
    scored = bm25_normalize(weights, popularity, k1=1.2, b=0.75)
    assert scored[0, 1] > scored[0, 2]
    flat = bm25_normalize(weights, popularity, k1=1.2, b=0.0)
    assert flat[0, 1] == pytest.approx(flat[0, 2])
