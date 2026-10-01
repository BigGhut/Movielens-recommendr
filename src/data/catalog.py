from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ItemCatalog:
    genres: list[str]
    item_genre: np.ndarray
    item_year: np.ndarray


@dataclass
class PairContexts:
    user_idx: np.ndarray
    item_idx: np.ndarray
    recent_genre: np.ndarray
    recent_year: np.ndarray
    recent_items: np.ndarray
    recent_mask: np.ndarray


@dataclass
class UserContexts:
    recent_genre: np.ndarray
    recent_year: np.ndarray
    recent_items: np.ndarray
    recent_mask: np.ndarray


def _year_from_title(title: str) -> float:
    if not isinstance(title, str) or len(title) < 6:
        return np.nan
    tail = title[-5:-1]
    return float(tail) if tail.isdigit() else np.nan


def build_item_catalog(movies_df: pd.DataFrame, item2idx: dict, train_df: pd.DataFrame) -> ItemCatalog:
    """Genre multi-hot and a train-scaled year for every indexed item."""
    genre_names = set()
    for raw in movies_df["genres"].fillna(""):
        for name in str(raw).split("|"):
            name = name.strip()
            if name and name != "(no genres listed)":
                genre_names.add(name)
    genres = sorted(genre_names)
    genre_index = {name: i for i, name in enumerate(genres)}
    num_items = len(item2idx)
    item_genre = np.zeros((num_items, len(genres)), dtype=np.float32)
    raw_year = np.full(num_items, np.nan, dtype=np.float64)

    for row in movies_df.itertuples(index=False):
        item_id = int(row.item_id)
        if item_id not in item2idx:
            continue
        idx = item2idx[item_id]
        for name in str(row.genres).split("|"):
            name = name.strip()
            if name in genre_index:
                item_genre[idx, genre_index[name]] = 1.0
        raw_year[idx] = _year_from_title(row.title)

    train_idx = [item2idx[int(item_id)] for item_id in train_df["item_id"].unique() if int(item_id) in item2idx]
    observed = raw_year[train_idx]
    year_mean = float(np.nanmean(observed)) if np.isfinite(observed).any() else 1995.0
    year_std = float(np.nanstd(observed)) if np.isfinite(observed).sum() > 1 else 1.0
    if not np.isfinite(year_std) or year_std < 1e-6:
        year_std = 1.0
    filled = np.where(np.isnan(raw_year), year_mean, raw_year)
    item_year = ((filled - year_mean) / year_std).astype(np.float32)
    return ItemCatalog(genres=genres, item_genre=item_genre, item_year=item_year)


def _append_recent(prev: list[int], catalog: ItemCatalog, k: int):
    if not prev:
        genre = np.zeros(catalog.item_genre.shape[1], dtype=np.float32)
        year = np.float32(0.0)
        items = np.zeros(k, dtype=np.int64)
        mask = np.zeros(k, dtype=np.float32)
        return genre, year, items, mask
    take = prev[-k:]
    genre = catalog.item_genre[take].mean(axis=0).astype(np.float32)
    year = np.float32(catalog.item_year[take].mean())
    items = np.zeros(k, dtype=np.int64)
    mask = np.zeros(k, dtype=np.float32)
    items[: len(take)] = take
    mask[: len(take)] = 1.0
    return genre, year, items, mask


def build_pair_contexts(
    train_df: pd.DataFrame,
    user2idx: dict,
    item2idx: dict,
    catalog: ItemCatalog,
    recent_k: int,
) -> PairContexts:
    """One row per train interaction. Recent context is movies rated strictly before it."""
    ordered = train_df.sort_values(["user_id", "timestamp"])
    users, items = [], []
    genres, years, recent_items, masks = [], [], [], []
    for user_id, group in ordered.groupby("user_id", sort=False):
        user_key = int(user_id)
        if user_key not in user2idx:
            continue
        seen: list[int] = []
        for item_id in group["item_id"]:
            item_key = int(item_id)
            if item_key not in item2idx:
                continue
            genre, year, recent, mask = _append_recent(seen, catalog, recent_k)
            users.append(user2idx[user_key])
            items.append(item2idx[item_key])
            genres.append(genre)
            years.append(year)
            recent_items.append(recent)
            masks.append(mask)
            seen.append(item2idx[item_key])
    return PairContexts(
        user_idx=np.asarray(users, dtype=np.int64),
        item_idx=np.asarray(items, dtype=np.int64),
        recent_genre=np.vstack(genres) if genres else np.zeros((0, catalog.item_genre.shape[1]), dtype=np.float32),
        recent_year=np.asarray(years, dtype=np.float32),
        recent_items=np.vstack(recent_items) if recent_items else np.zeros((0, recent_k), dtype=np.int64),
        recent_mask=np.vstack(masks) if masks else np.zeros((0, recent_k), dtype=np.float32),
    )


def build_user_contexts(
    train_df: pd.DataFrame,
    user2idx: dict,
    item2idx: dict,
    catalog: ItemCatalog,
    recent_k: int,
) -> UserContexts:
    """Last `recent_k` train movies for every user. This is the query context at serve time."""
    num_users = len(user2idx)
    num_genres = catalog.item_genre.shape[1]
    recent_genre = np.zeros((num_users, num_genres), dtype=np.float32)
    recent_year = np.zeros(num_users, dtype=np.float32)
    recent_items = np.zeros((num_users, recent_k), dtype=np.int64)
    recent_mask = np.zeros((num_users, recent_k), dtype=np.float32)
    ordered = train_df.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id", sort=False):
        user_key = int(user_id)
        if user_key not in user2idx:
            continue
        seen = [item2idx[int(item_id)] for item_id in group["item_id"] if int(item_id) in item2idx]
        genre, year, recent, mask = _append_recent(seen, catalog, recent_k)
        row = user2idx[user_key]
        recent_genre[row] = genre
        recent_year[row] = year
        recent_items[row] = recent
        recent_mask[row] = mask
    return UserContexts(recent_genre, recent_year, recent_items, recent_mask)


def popularity_weights(train_df: pd.DataFrame, item2idx: dict) -> np.ndarray:
    weights = np.ones(len(item2idx), dtype=np.float64)
    for item_id, count in train_df["item_id"].value_counts().items():
        key = int(item_id)
        if key in item2idx:
            weights[item2idx[key]] += float(count)
    return weights
