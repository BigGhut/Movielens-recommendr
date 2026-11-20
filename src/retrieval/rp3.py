"""Optional RP3beta candidates. The serving path stays off unless the config says so.

Co-visitation is not a serving source. The offline bench still measures it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retrieval.graph_sources import (
    build_interaction_matrix,
    rp3_similarity,
    score_users,
    top_indices,
)

SOURCE_COLUMNS = (
    "tt_score",
    "tt_rank",
    "tt_flag",
    "rp3_score",
    "rp3_rank",
    "rp3_flag",
)


def rp3_enabled_settings(config: dict | None) -> dict | None:
    """Return alpha, beta and top_k, or None when the flag is missing or false."""
    retrieval = (config or {}).get("retrieval") or {}
    rp3 = retrieval.get("rp3") or {}
    if not bool(rp3.get("enabled", False)):
        return None
    pipeline = (config or {}).get("pipeline") or {}
    return {
        "alpha": float(rp3.get("alpha", 1.0)),
        "beta": float(rp3.get("beta", 0.7)),
        "top_k": int(rp3.get("top_k", pipeline.get("retrieval_top_k", 200))),
    }


class TrainRP3:
    """One item-user-item hop built from the train slice only."""

    def __init__(
        self,
        interactions,
        similarity: np.ndarray,
        item2idx: dict,
        idx2item: np.ndarray,
        top_k: int,
    ):
        self.interactions = interactions
        self.similarity = similarity
        self.item2idx = item2idx
        self.idx2item = idx2item
        self.top_k = int(top_k)

    @classmethod
    def from_frame(
        cls,
        train_df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        alpha: float,
        beta: float,
        top_k: int,
    ) -> TrainRP3:
        interactions, popularity = build_interaction_matrix(train_df, user2idx, item2idx)
        similarity = rp3_similarity(interactions, popularity, alpha, beta)
        idx2item = np.empty(len(item2idx), dtype=np.int32)
        for item_id, index in item2idx.items():
            idx2item[int(index)] = int(item_id)
        return cls(interactions, similarity, item2idx, idx2item, top_k)

    def score_row(self, user_index: int, history: set[int]) -> np.ndarray:
        """Scores for one user. Train items and the caller's history are -inf."""
        scores = score_users(
            self.interactions,
            self.similarity,
            np.array([user_index], dtype=np.int32),
        )[0]
        for item_id in history:
            index = self.item2idx.get(int(item_id))
            if index is not None:
                scores[int(index)] = -np.inf
        return scores

    def top_ids(self, scores: np.ndarray) -> list[int]:
        values, indices = top_indices(scores.reshape(1, -1), self.top_k)
        item_ids = []
        for value, index in zip(values[0], indices[0], strict=True):
            if not np.isfinite(value):
                continue
            item_ids.append(int(self.idx2item[int(index)]))
        return item_ids


def union_with_rp3(
    tower_ids: list[int],
    rp3_ids: list[int],
    top_k: int,
) -> tuple[list[int], list[float]]:
    """Tower order first. An RP3-only item gets retrieval rank ``top_k``."""
    chosen = [int(item_id) for item_id in tower_ids]
    ranks = [float(position) for position in range(len(chosen))]
    seen = set(chosen)
    outside = float(top_k)
    for item_id in rp3_ids:
        item_id = int(item_id)
        if item_id in seen:
            continue
        chosen.append(item_id)
        ranks.append(outside)
        seen.add(item_id)
    return chosen, ranks


def two_tower_score_row(
    user_emb: np.ndarray,
    faiss_index,
    n_items: int,
    history: set[int],
    item2idx: dict,
) -> np.ndarray:
    """Full-catalog FAISS scores. History items are -inf and sort last."""
    query = user_emb.reshape(1, -1) if user_emb.ndim == 1 else user_emb
    distances, indices = faiss_index.search(query, n_items)
    scores = np.full(n_items, -np.inf, dtype=np.float64)
    for distance, raw_index in zip(distances[0], indices[0], strict=True):
        index = int(raw_index)
        if 0 <= index < n_items:
            scores[index] = float(distance)
    for item_id in history:
        index = item2idx.get(int(item_id))
        if index is not None:
            scores[int(index)] = -np.inf
    return scores


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    rank = np.empty(scores.shape[0], dtype=np.int32)
    rank[order] = np.arange(scores.shape[0], dtype=np.int32)
    return rank


def source_feature_columns(
    item_ids: list[int],
    tower_ids: list[int],
    rp3_ids: list[int],
    tt_scores: np.ndarray,
    rp3_scores: np.ndarray,
    item2idx: dict,
) -> dict[str, list[float]]:
    """Score, full-catalog rank and pool flag for the two sources."""
    tt_rank = _ranks(tt_scores)
    rp3_rank = _ranks(rp3_scores)
    tower = {int(item_id) for item_id in tower_ids}
    pool = {int(item_id) for item_id in rp3_ids}
    columns = {name: [] for name in SOURCE_COLUMNS}
    for item_id in item_ids:
        index = int(item2idx[int(item_id)])
        tt_value = tt_scores[index]
        rp3_value = rp3_scores[index]
        columns["tt_score"].append(float(tt_value) if np.isfinite(tt_value) else 0.0)
        columns["tt_rank"].append(float(tt_rank[index]))
        columns["tt_flag"].append(1.0 if int(item_id) in tower else 0.0)
        columns["rp3_score"].append(float(rp3_value) if np.isfinite(rp3_value) else 0.0)
        columns["rp3_rank"].append(float(rp3_rank[index]))
        columns["rp3_flag"].append(1.0 if int(item_id) in pool else 0.0)
    return columns


def trained_source_columns(ranker) -> list[str]:
    """Source columns the loaded ranker actually learned. The published model has none."""
    model = getattr(ranker, "model", None)
    names = set(getattr(model, "feature_names_", None) or [])
    return [column for column in SOURCE_COLUMNS if column in names]
