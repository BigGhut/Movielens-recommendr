from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.train_ranker import build_labeled_pairs
from src.pipeline.recommend import RecommendationPipeline, _build_rp3
from src.retrieval.index import FAISSIndex
from src.retrieval.rp3 import SOURCE_COLUMNS, rp3_enabled_settings, union_with_rp3


def test_config_keeps_rp3_off_and_settings_default():
    config = yaml.safe_load(Path("configs/model_config.yaml").read_text(encoding="utf-8"))
    assert config["retrieval"]["rp3"]["enabled"] is False
    assert rp3_enabled_settings(config) is None
    assert "covis" not in config["retrieval"]
    assert _build_rp3(config, pd.DataFrame({"user_id": [1], "item_id": [2]}), {1: 0}, {2: 0}) is None

    enabled = {"pipeline": {"retrieval_top_k": 50}, "retrieval": {"rp3": {"enabled": True}}}
    assert rp3_enabled_settings(enabled) == {"alpha": 1.0, "beta": 0.7, "top_k": 50}
    with pytest.raises(ValueError, match="train_df"):
        _build_rp3(enabled, None, {1: 0}, {2: 0})


def test_union_keeps_tower_order_and_marks_rp3_only_items():
    item_ids, ranks = union_with_rp3([11, 10], [10, 13], 200)
    assert item_ids == [11, 10, 13]
    assert ranks == [0.0, 1.0, 200.0]


def _catalog():
    vectors = np.array(
        [
            [0.8, 0.6, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = FAISSIndex(embedding_dim=4)
    index.build(vectors)
    train = pd.DataFrame(
        {
            "user_id": [10, 10, 11, 11],
            "item_id": [100, 101, 101, 102],
            "rating": [5, 5, 5, 5],
            "timestamp": [1, 2, 3, 4],
        }
    )
    movies = pd.DataFrame(
        {
            "item_id": [100, 101, 102, 103],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Drama", "Drama", "Drama"],
        }
    )
    return {
        "index": index,
        "vectors": vectors,
        "train": train,
        "movies": movies,
        "user2idx": {10: 0, 11: 1},
        "item2idx": {100: 0, 101: 1, 102: 2, 103: 3},
        "idx2item": {0: 100, 1: 101, 2: 102, 3: 103},
    }


class _Ranker:
    def __init__(self, names):
        self.model = type("Model", (), {"feature_names_": list(names)})()
        self.frame = None

    def predict(self, frame):
        self.frame = frame.copy()
        return np.zeros(len(frame), dtype=np.float64)


def _pipeline(catalog, enabled, names):
    config = {"retrieval_top_k": 1}
    if enabled:
        config["retrieval"] = {"rp3": {"enabled": True, "alpha": 1.0, "beta": 0.0, "top_k": 1}}
    user_features = pd.DataFrame({"user_id": [10], "avg_rating": [4.0]})
    item_features = pd.DataFrame(
        {
            "item_id": [100, 101, 102, 103],
            "item_avg_rating": [4.0, 4.0, 4.0, 4.0],
            "genres": ["Drama", "Drama", "Drama", "Drama"],
        }
    )
    ranker = _Ranker(names)
    pipeline = RecommendationPipeline(
        user_embs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        item_embs=catalog["vectors"],
        faiss_index=catalog["index"],
        catboost_ranker=ranker,
        user_features=user_features,
        item_features=item_features,
        user2idx=catalog["user2idx"],
        item2idx=catalog["item2idx"],
        idx2item=catalog["idx2item"],
        movies_df=catalog["movies"],
        ratings_df=catalog["train"],
        config=config,
        train_df=catalog["train"] if enabled else None,
    )
    pipeline.ranker_mode = "blend"
    pipeline.ranker_beta = 0.0
    return pipeline, ranker


def test_flag_off_keeps_the_faiss_list_and_on_unions_rp3():
    catalog = _catalog()
    off, _ranker = _pipeline(catalog, enabled=False, names=[])
    assert [item["item_id"] for item in off.recommend(10)] == [103]

    on, ranker = _pipeline(catalog, enabled=True, names=[])
    assert [item["item_id"] for item in on.recommend(10)] == [103, 102]
    assert "rp3_score" not in ranker.frame.columns

    named, ranker = _pipeline(catalog, enabled=True, names=SOURCE_COLUMNS)
    assert [item["item_id"] for item in named.recommend(10)] == [103, 102]
    assert ranker.frame["tt_flag"].tolist() == [1.0, 0.0]
    assert ranker.frame["rp3_flag"].tolist() == [0.0, 1.0]
    assert "retrieval_rank" not in ranker.frame.columns


class _StubRP3:
    def __init__(self):
        self.item2idx = {10: 0, 11: 1, 12: 2, 13: 3}

    def score_row(self, user_index, history):
        del user_index, history
        return np.array([0.1, 0.4, -np.inf, 2.0], dtype=np.float64)

    def top_ids(self, scores):
        del scores
        return [13]


def test_training_pairs_grow_only_when_rp3_is_passed():
    vectors = np.array(
        [
            [0.6, 0.8, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = FAISSIndex(embedding_dim=4)
    index.build(vectors)
    history = pd.DataFrame({"user_id": [7], "item_id": [12]})
    labels = pd.DataFrame({"user_id": [7], "item_id": [13], "rating": [5]})
    args = (
        history,
        labels,
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        {7: 0},
        {0: 10, 1: 11, 2: 12, 3: 13},
        index,
    )

    alone, alone_labels = build_labeled_pairs(*args, top_k=2, positive_threshold=4)
    assert alone.empty
    assert alone_labels.size == 0

    pairs, relevance = build_labeled_pairs(*args, top_k=2, positive_threshold=4, rp3=_StubRP3())
    assert pairs["item_id"].tolist() == [11, 10, 13]
    assert pairs["retrieval_rank"].tolist() == [0.0, 1.0, 2.0]
    assert pairs["tt_flag"].tolist() == [1.0, 1.0, 0.0]
    assert pairs["rp3_flag"].tolist() == [0.0, 0.0, 1.0]
    assert relevance.tolist() == [0.0, 0.0, 1.0]
