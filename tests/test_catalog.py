import numpy as np
import pandas as pd

from src.data.catalog import build_item_catalog, build_pair_contexts, build_user_contexts
from src.data.feature_store import build_pair_features, build_recent_centroids


def _movies():
    return pd.DataFrame({
        "item_id": [10, 11, 12],
        "title": ["A (1990)", "B (2000)", "C (2010)"],
        "genres": ["Action", "Action|Comedy", "Drama"],
    })


def _train():
    return pd.DataFrame({
        "user_id": [7, 7, 7],
        "item_id": [10, 11, 12],
        "rating": [5, 4, 5],
        "timestamp": [1, 2, 3],
    })


def test_pair_history_excludes_the_current_item():
    movies = _movies()
    train = _train()
    item2idx = {10: 0, 11: 1, 12: 2}
    user2idx = {7: 0}
    catalog = build_item_catalog(movies, item2idx, train)
    contexts = build_pair_contexts(train, user2idx, item2idx, catalog, recent_k=2)

    assert contexts.item_idx.tolist() == [0, 1, 2]
    assert contexts.recent_mask[0].tolist() == [0.0, 0.0]
    assert contexts.recent_items[1, 0] == 0
    assert contexts.recent_mask[1].tolist() == [1.0, 0.0]
    assert contexts.recent_items[2].tolist() == [0, 1]
    assert contexts.recent_mask[2].tolist() == [1.0, 1.0]
    assert catalog.item_genre.shape == (3, 3)


def test_user_context_keeps_the_last_movies():
    movies = _movies()
    train = _train()
    item2idx = {10: 0, 11: 1, 12: 2}
    user2idx = {7: 0}
    catalog = build_item_catalog(movies, item2idx, train)
    users = build_user_contexts(train, user2idx, item2idx, catalog, recent_k=2)
    assert users.recent_items[0].tolist() == [1, 2]
    assert users.recent_mask[0].tolist() == [1.0, 1.0]


def test_recent_neighbor_prefers_the_latest_movie():
    item_embs = np.eye(2, dtype=np.float32)
    history = pd.DataFrame({
        "user_id": [7, 7],
        "item_id": [10, 11],
        "timestamp": [1, 2],
    })
    user2idx = {7: 0}
    item2idx = {10: 0, 11: 1}
    centroids = build_recent_centroids(history, item_embs, user2idx, item2idx, recent_k=1)
    assert np.allclose(centroids[0], item_embs[1])

    user_feats = pd.DataFrame({"user_id": [7], "avg_rating": [4.0]})
    item_feats = pd.DataFrame({
        "item_id": [10, 11],
        "item_avg_rating": [3.0, 5.0],
        "genres": ["Action", "Comedy"],
    })
    pairs = pd.DataFrame({"user_id": [7, 7], "item_id": [10, 11]})
    frame = build_pair_features(
        user_feats, item_feats, item_embs, item_embs, user2idx, item2idx, pairs,
        user_recent_embs=centroids,
    )
    scores = frame["recent_neighbor_score"].tolist()
    assert scores[1] > scores[0]
