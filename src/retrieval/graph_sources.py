"""Train-only item-item candidate sources.

RP3beta is one item-user-item hop, not a second hop on the resulting
item graph. With a binary train interaction matrix A:

    P_ui[u, i] = A[u, i] / deg(u)
    P_iu[i, u] = A[u, i] / deg(i)
    S = (P_iu ** alpha) @ (P_ui ** alpha)
    S[:, j] /= deg(j) ** beta

The power is elementwise, on the transition probabilities. The diagonal
is cleared. A user score is the sum of S over items in that user's train
history.

Co-visitation counts every pair of distinct train items inside one user
history. A pair at timestamps t_i and t_j gets weight exp(-|t_i - t_j| / tau).
BM25-style popularity normalization then saturates that weight:

    pop(j) is the number of train interactions of the destination item
    norm(j) = 1 - b + b * pop(j) / mean(pop)
    BM25[i, j] = tf[i, j] * (k1 + 1) / (tf[i, j] + k1 * norm(j))

Neither builder looks at validation or test. Callers have to pass the
train frame only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def build_interaction_matrix(
    train_df: pd.DataFrame,
    user2idx: dict,
    item2idx: dict,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Binary user-item matrix aligned with the two id maps.

    Repeated pairs stay a single edge. The returned popularity vector is
    the column sum, one entry per item index.
    """
    users = train_df["user_id"].map(user2idx)
    items = train_df["item_id"].map(item2idx)
    valid = users.notna() & items.notna()
    rows = users[valid].to_numpy(dtype=np.int32)
    cols = items[valid].to_numpy(dtype=np.int32)
    data = np.ones(rows.shape[0], dtype=np.float64)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(user2idx), len(item2idx)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    csr = matrix.tocsr()
    popularity = np.asarray(csr.sum(axis=0)).ravel()
    return csr, popularity


def rp3_similarity(
    interactions: sparse.csr_matrix,
    popularity: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Item-item RP3beta. Shape is (n_items, n_items)."""
    user_degree = np.asarray(interactions.sum(axis=1)).ravel()
    user_scale = _inverse_power(user_degree, alpha)
    item_scale = _inverse_power(popularity, alpha)
    user_to_item = sparse.diags(user_scale) @ interactions
    item_to_user = sparse.diags(item_scale) @ interactions.T.tocsr()
    similarity = (item_to_user @ user_to_item).toarray()
    np.fill_diagonal(similarity, 0.0)
    similarity *= _inverse_power(popularity, beta)
    return similarity


def rp3_hop(interactions: sparse.csr_matrix, popularity: np.ndarray, alpha: float) -> np.ndarray:
    """The alpha hop before the beta column scaling, so a grid can reuse it."""
    return rp3_similarity(interactions, popularity, alpha, 0.0)


def apply_popularity_beta(similarity: np.ndarray, popularity: np.ndarray, beta: float) -> np.ndarray:
    """Divide destination columns by popularity ** beta. beta=0 is a copy."""
    scaled = np.array(similarity, dtype=np.float64, copy=True)
    scaled *= _inverse_power(popularity, beta)
    return scaled


def covisitation_weights(
    train_df: pd.DataFrame,
    item2idx: dict,
    tau_seconds: float,
) -> np.ndarray:
    """Symmetric time-decayed co-occurrence of train items. Diagonal is zero."""
    frame = train_df.loc[:, ["user_id", "item_id", "timestamp"]].copy()
    frame["item_idx"] = frame["item_id"].map(item2idx)
    frame = frame.dropna(subset=["item_idx"])
    frame["item_idx"] = frame["item_idx"].astype(np.int32)
    frame = frame.sort_values(["user_id", "timestamp", "item_idx"], kind="mergesort")
    frame = frame.drop_duplicates(["user_id", "item_idx"], keep="last")

    item_idx = frame["item_idx"].to_numpy()
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    user_ids = frame["user_id"].to_numpy()
    cuts = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(user_ids)]))

    weights = np.zeros((len(item2idx), len(item2idx)), dtype=np.float64)
    tau = float(tau_seconds)
    for start, end in zip(starts, ends, strict=True):
        idx = item_idx[start:end]
        size = int(idx.shape[0])
        if size < 2:
            continue
        delta = np.abs(timestamps[start:end, None] - timestamps[None, start:end])
        pair = np.exp(-delta / tau)
        np.fill_diagonal(pair, 0.0)
        left = np.repeat(idx, size)
        right = np.tile(idx, size)
        np.add.at(weights, (left, right), pair.ravel())
    np.fill_diagonal(weights, 0.0)
    return weights


def bm25_normalize(weights: np.ndarray, popularity: np.ndarray, k1: float, b: float) -> np.ndarray:
    """BM25 TF saturation. Popularity is the destination item's train degree."""
    positive = popularity[popularity > 0]
    average = float(positive.mean()) if positive.size else 1.0
    length = 1.0 - b + b * popularity / average
    length = np.maximum(length, 1e-12)
    scored = weights * (k1 + 1.0) / (weights + k1 * length[None, :])
    scored = np.where(weights > 0.0, scored, 0.0)
    np.fill_diagonal(scored, 0.0)
    return scored


def score_users(interactions: sparse.csr_matrix, similarity: np.ndarray, user_rows: np.ndarray) -> np.ndarray:
    """Sum similarity over each user's train items. Seen items become -inf."""
    block = interactions[user_rows]
    scores = np.asarray(block @ similarity, dtype=np.float64)
    coo = block.tocoo()
    scores[coo.row, coo.col] = -np.inf
    return scores


def top_indices(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest k scores per row. Values and indices have shape (n_rows, k)."""
    _n_rows, n_cols = scores.shape
    width = min(int(k), n_cols)
    chosen = np.argpartition(scores, n_cols - width, axis=1)[:, n_cols - width :]
    chosen_scores = np.take_along_axis(scores, chosen, axis=1)
    order = np.argsort(-chosen_scores, axis=1, kind="mergesort")
    indices = np.take_along_axis(chosen, order, axis=1)
    values = np.take_along_axis(chosen_scores, order, axis=1)
    return values, indices


def indices_to_item_ids(indices: np.ndarray, values: np.ndarray, idx2item: np.ndarray) -> np.ndarray:
    """Map item indices to ids. Non-finite scores become -1."""
    item_ids = idx2item[indices]
    item_ids = item_ids.astype(np.int32, copy=True)
    item_ids[~np.isfinite(values)] = -1
    return item_ids


def _inverse_power(degree: np.ndarray, power: float) -> np.ndarray:
    """degree ** -power, with zeros staying zero. power=0 yields ones on positive degrees."""
    scale = np.zeros(degree.shape, dtype=np.float64)
    positive = degree > 0
    if power == 0.0:
        scale[positive] = 1.0
        return scale
    scale[positive] = np.power(degree[positive], -power)
    return scale
