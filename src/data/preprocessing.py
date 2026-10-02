"""MovieLens loaders.

`temporal_split` is the training and evaluation split. The rating column is
`item_id`. Users with fewer than `min_ratings` are dropped, and the order is
`(user_id, timestamp)` only. Published metrics use this function.

`split_ratings_by_time` is the CSV pipeline split. The rating column is
`movie_id`. The function does not filter users. The order is
`(user_id, timestamp, movie_id)`, so equal timestamps break on `movie_id`.
`preprocess_and_save` writes train/val/test.csv from that split.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Загрузка рейтингов."""
    file_path = data_dir / "ratings.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Загрузка фильмов."""
    file_path = data_dir / "movies.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )


def load_users(data_dir: Path) -> pd.DataFrame:
    """Загрузка пользователей."""
    file_path = data_dir / "users.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )


def temporal_split(
    ratings: pd.DataFrame, min_ratings: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Выполняет строгое темпоральное разделение данных:
    последнее взаимодействие пользователя -> test,
    предпоследнее -> validation,
    остальные -> train.
    Фильтрует пользователей с < min_ratings.
    """
    user_counts = ratings.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    filtered_ratings.sort_values(["user_id", "timestamp"], inplace=True)

    test_df = filtered_ratings.groupby("user_id").tail(1)
    rest_df = filtered_ratings.drop(test_df.index)
    val_df = rest_df.groupby("user_id").tail(1)
    train_df = rest_df.drop(val_df.index)

    return train_df, val_df, test_df


def create_id_mappings(ratings: pd.DataFrame) -> tuple[dict, dict]:
    """
    Создает словари для перевода реальных user_id и item_id в непрерывные индексы (начиная с 0).
    """
    unique_users = sorted(ratings["user_id"].unique())
    unique_items = sorted(ratings["item_id"].unique())

    user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item2idx = {item_id: idx for idx, item_id in enumerate(unique_items)}

    return user2idx, item2idx


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw MovieLens-1M files. Rating column is movie_id, user zip is zip_code."""
    ratings_path = raw_dir / "ratings.dat"
    users_path = raw_dir / "users.dat"
    movies_path = raw_dir / "movies.dat"

    for path in [ratings_path, users_path, movies_path]:
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Raw data file is empty: {path}")

    logger.info("Loading raw MovieLens-1M datasets...")

    try:
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )
        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
    except (OSError, ValueError) as exc:
        logger.error("Error reading raw data files: %s", exc)
        raise ValueError(f"Failed to parse raw data files: {exc}") from exc

    if ratings.empty:
        raise ValueError("Ratings DataFrame is empty.")
    if users.empty:
        raise ValueError("Users DataFrame is empty.")
    if movies.empty:
        raise ValueError("Movies DataFrame is empty.")

    return ratings, users, movies


def filter_users(ratings: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Filter out users with fewer than min_interactions."""
    if ratings.empty:
        raise ValueError("Cannot filter empty ratings DataFrame.")

    logger.info("Filtering users with fewer than %s interactions...", min_interactions)
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    logger.info(
        "Filtered out %s users. Remaining users: %s. Remaining ratings: %s.",
        len(user_counts) - len(valid_users),
        len(valid_users),
        len(filtered_ratings),
    )
    return filtered_ratings


def split_ratings_by_time(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CSV-pipeline split. Does not filter users. Ties break on movie_id.

    For each user the last row is test, the second-to-last is validation,
    and the rest is train. This is not `temporal_split`: that function filters
    on `item_id` frames and does not use `movie_id` as a tie-break.
    """
    if ratings.empty:
        raise ValueError("Cannot split empty ratings DataFrame.")

    logger.info("Performing chronological temporal split per user...")

    sorted_ratings = ratings.sort_values(by=["user_id", "timestamp", "movie_id"]).reset_index(drop=True)

    grouped = sorted_ratings.groupby("user_id")
    group_size = grouped["user_id"].transform("count")
    cum_count = grouped.cumcount()

    if (group_size < 3).any():
        logger.warning("Some users have fewer than 3 interactions. They will have empty train set.")

    train_mask = cum_count < (group_size - 2)
    val_mask = cum_count == (group_size - 2)
    test_mask = cum_count == (group_size - 1)

    train = sorted_ratings[train_mask].copy()
    val = sorted_ratings[val_mask].copy()
    test = sorted_ratings[test_mask].copy()

    logger.info("Split results - Train: %s, Val: %s, Test: %s", len(train), len(val), len(test))
    return train, val, test


def preprocess_and_save(raw_dir: Path, processed_dir: Path, min_interactions: int = 5) -> None:
    """Load, filter, split, merge, and save MovieLens-1M datasets."""
    ratings, users, movies = load_raw_data(raw_dir)

    filtered_ratings = filter_users(ratings, min_interactions)
    if filtered_ratings.empty:
        raise ValueError("No ratings remaining after user filtering.")

    train, val, test = split_ratings_by_time(filtered_ratings)

    logger.info("Merging user and movie metadata with splits...")

    def merge_metadata(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="movie_id", how="left")
        return df

    train_merged = merge_metadata(train)
    val_merged = merge_metadata(val)
    test_merged = merge_metadata(test)

    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving splits to %s...", processed_dir)
    train_merged.to_csv(processed_dir / "train.csv", index=False)
    val_merged.to_csv(processed_dir / "val.csv", index=False)
    test_merged.to_csv(processed_dir / "test.csv", index=False)

    logger.info("Preprocessing complete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write train/val/test.csv from MovieLens-1M dat files.")
    parser.add_argument("--raw-dir", default="data/raw/ml-1m")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--min-interactions", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    preprocess_and_save(Path(args.raw_dir), Path(args.processed_dir), args.min_interactions)


if __name__ == "__main__":
    main()
