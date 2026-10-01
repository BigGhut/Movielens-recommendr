import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw MovieLens-1M datasets from the specified directory.
    
    Args:
        raw_dir: Path to the directory containing raw MovieLens-1M files.
        
    Returns:
        Tuple of (ratings, users, movies) DataFrames.
    """
    ratings_path = raw_dir / "ratings.dat"
    users_path = raw_dir / "users.dat"
    movies_path = raw_dir / "movies.dat"

    # Check that all raw files exist
    for path in [ratings_path, users_path, movies_path]:
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Raw data file is empty: {path}")

    logger.info("Loading raw MovieLens-1M datasets...")
    
    try:
        # MovieLens-1M files use '::' separator and are typically latin-1 or ISO-8859-1 encoded.
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1"
        )
        
        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1"
        )
        
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1"
        )
    except Exception as e:
        logger.error(f"Error reading raw data files: {e}")
        raise ValueError(f"Failed to parse raw data files: {e}")

    if ratings.empty:
        raise ValueError("Ratings DataFrame is empty.")
    if users.empty:
        raise ValueError("Users DataFrame is empty.")
    if movies.empty:
        raise ValueError("Movies DataFrame is empty.")

    return ratings, users, movies

def filter_users(ratings: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    """Filter out users with fewer than min_interactions.
    
    Args:
        ratings: Ratings DataFrame.
        min_interactions: Minimum number of ratings per user.
        
    Returns:
        Filtered ratings DataFrame.
    """
    if ratings.empty:
        raise ValueError("Cannot filter empty ratings DataFrame.")
        
    logger.info(f"Filtering users with fewer than {min_interactions} interactions...")
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()
    
    logger.info(f"Filtered out {len(user_counts) - len(valid_users)} users. "
                f"Remaining users: {len(valid_users)}. Remaining ratings: {len(filtered_ratings)}.")
    return filtered_ratings

def temporal_split(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a deterministic temporal train/val/test split for each user.
    
    For each user, ratings are sorted chronologically by ['user_id', 'timestamp', 'movie_id'].
    The last rating is assigned to the test set, the second-to-last is assigned to the
    validation set, and all preceding ratings are assigned to the training set.
    
    Args:
        ratings: Ratings DataFrame.
        
    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    if ratings.empty:
        raise ValueError("Cannot split empty ratings DataFrame.")
        
    logger.info("Performing chronological temporal split per user...")
    
    # Sort deterministically
    sorted_ratings = ratings.sort_values(
        by=["user_id", "timestamp", "movie_id"]
    ).reset_index(drop=True)
    
    # Calculate sizes and cumulative count within each user group
    grouped = sorted_ratings.groupby("user_id")
    group_size = grouped["user_id"].transform("count")
    cum_count = grouped.cumcount()
    
    # Verify that all users have at least 3 ratings (to allow 1 val, 1 test, and >=1 train)
    # Note: our filter_users ensures min_interactions >= 5, so this is guaranteed.
    if (group_size < 3).any():
        logger.warning("Some users have fewer than 3 interactions. They will have empty train set.")
        
    train_mask = cum_count < (group_size - 2)
    val_mask = cum_count == (group_size - 2)
    test_mask = cum_count == (group_size - 1)
    
    train = sorted_ratings[train_mask].copy()
    val = sorted_ratings[val_mask].copy()
    test = sorted_ratings[test_mask].copy()
    
    logger.info(f"Split results - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def preprocess_and_save(raw_dir: Path, processed_dir: Path, min_interactions: int = 5) -> None:
    """Load, filter, split, merge, and save MovieLens-1M datasets."""
    # Load raw datasets
    ratings, users, movies = load_raw_data(raw_dir)
    
    # Filter users
    filtered_ratings = filter_users(ratings, min_interactions)
    if filtered_ratings.empty:
        raise ValueError("No ratings remaining after user filtering.")
        
    # Split chronologically
    train, val, test = temporal_split(filtered_ratings)
    
    # Merge metadata
    logger.info("Merging user and movie metadata with splits...")
    
    # Helper to merge metadata
    def merge_metadata(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="movie_id", how="left")
        return df
        
    train_merged = merge_metadata(train)
    val_merged = merge_metadata(val)
    test_merged = merge_metadata(test)
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    
    logger.info(f"Saving splits to {processed_dir}...")
    train_merged.to_csv(train_path, index=False)
    val_merged.to_csv(val_path, index=False)
    test_merged.to_csv(test_path, index=False)
    
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    raw_path = Path("data/raw/ml-1m")
    processed_path = Path("data/processed")
    preprocess_and_save(raw_path, processed_path)
