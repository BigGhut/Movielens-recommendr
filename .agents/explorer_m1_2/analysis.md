# Milestone 1: Data Pipeline Analysis & Design Report

This report presents the analysis, architectural design, and file templates for the MovieLens-1M data pipeline (Milestone 1) of the Two-Tower Recommendation System.

---

## 1. Data Ingestion & Storage Strategy

### 1.1 Source Dataset Details
- **Source URL**: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- **Size**: ~6 MB compressed, ~24 MB uncompressed.
- **Files of Interest**:
  - `ratings.dat`: UserID::MovieID::Rating::Timestamp
  - `users.dat`: UserID::Gender::Age::Occupation::Zip-code
  - `movies.dat`: MovieID::Title::Genres

### 1.2 Ingestion Challenges & Solutions
1. **Multi-Character Separator (`::`)**: Standard pandas CSV parser engine (`c`) does not support multi-character delimiters. We must use `engine='python'` with `sep='::'` in `pd.read_csv`.
2. **File Encoding**: The files contain non-UTF-8 characters (specifically movie titles in `movies.dat`). Using `encoding='ISO-8859-1'` (latin-1) prevents encoding errors.
3. **No Header Rows**: The `.dat` files do not contain headers. Column names must be supplied explicitly during parsing.
4. **Network Access**: The downloader script must be runnable locally on the user's machine using Python's standard libraries (`urllib.request` and `zipfile`), ensuring no third-party package dependencies for downloading.

---

## 2. Preprocessing & Splitting Pipeline

### 2.1 User Interaction Filtering
To ensure robust training of user and item embeddings and prevent cold-start noise:
- Filter out users who have **fewer than 5 ratings** in total.
- This is performed *before* splitting to ensure that after splitting, every user has at least 3 train interactions, 1 validation interaction, and 1 test interaction.

### 2.2 Temporal Split Logic
We apply a **local temporal split** per user (commonly referred to as leave-last-two-out split):
1. For each user, sort all their interactions chronologically by `timestamp` ascending.
2. If multiple interactions share the same timestamp (e.g., bulk ratings in the same second), sort them by `movie_id` ascending as a deterministic tie-breaker.
3. Split the sorted interactions as follows:
   - **Test Set**: The latest interaction (index `-1`).
   - **Validation Set**: The second-to-latest interaction (index `-2`).
   - **Train Set**: All remaining interactions (indices `0` to `-3`).

This ensures:
- **No temporal leakage**: No rating in the test set occurs before any rating in the train or validation sets for the same user.
- **Consistent user representation**: Every user is represented across all three splits, enabling the model to learn user embeddings on the training set and evaluate them on validation/test sets.

### 2.3 Vectorized Split Implementation
Iterating over users in Python is highly inefficient. We use a vectorized Pandas algorithm using cumulative counts:
1. Sort the combined DataFrame by `["user_id", "timestamp", "movie_id"]`.
2. Calculate the interaction index per user from the beginning using `.groupby("user_id").cumcount()`.
3. Compute the reverse rank per user: `reverse_rank = total_interactions - 1 - interaction_index`.
4. Slice the DataFrame: `reverse_rank == 0` for test, `reverse_rank == 1` for validation, and `reverse_rank >= 2` for train.

---

## 3. Directory Layout

The proposed directory layout for the data pipeline is as follows:

```text
z:/pet-project/recsys-two-tower/
├── data/
│   ├── raw/                  # Downloaded raw ml-1m files (gitignored except placeholder)
│   └── processed/            # Processed CSV splits (train.csv, val.csv, test.csv)
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── download.py       # Dataset downloader and extractor
│       └── preprocess.py     # Ingestion, filtering, splitting, and merging
└── tests/
    ├── __init__.py
    └── test_data_pipeline.py # Unit tests verifying split logic and leakage prevention
```

---

## 4. File Templates

Here are the complete, production-ready Python file templates containing strict type hinting, docstrings, and robust error handling.

### 4.1 Downloader Script: `src/data/download.py`

```python
import os
import zipfile
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_and_extract(dest_dir: str = "data/raw") -> Path:
    """
    Downloads the MovieLens-1M dataset and extracts it to the target directory.
    
    Args:
        dest_dir: Destination directory path for raw data.
        
    Returns:
        Path to the extracted dataset directory.
    """
    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    
    zip_file_path = dest_path / "ml-1m.zip"
    extracted_dir = dest_path / "ml-1m"
    
    # 1. Download Zip File
    if not zip_file_path.exists() and not extracted_dir.exists():
        logger.info(f"Downloading MovieLens-1M dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, zip_file_path)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset from {DATA_URL}: {e}")
            raise e
    else:
        logger.info("Zip file or extracted directory already exists. Skipping download.")
        
    # 2. Extract Zip File
    if not extracted_dir.exists():
        logger.info(f"Extracting {zip_file_path} to {dest_path}...")
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(dest_path)
            logger.info("Extraction completed successfully.")
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}")
            raise e
        
        # Clean up zip file after successful extraction to save space
        if zip_file_path.exists():
            os.remove(zip_file_path)
            logger.info("Cleaned up download zip file.")
    else:
        logger.info(f"Extracted directory {extracted_dir} already exists. Skipping extraction.")
        
    # 3. Quick Sanity Check
    required_files = ["ratings.dat", "users.dat", "movies.dat"]
    for f in required_files:
        file_path = extracted_dir / f
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file {f} was not found in extracted directory.")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Extracted file {f} is empty.")
            
    logger.info("MovieLens-1M dataset is downloaded and verified.")
    return extracted_dir

if __name__ == "__main__":
    download_and_extract()
```

### 4.2 Preprocessing Script: `src/data/preprocess.py`

```python
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data(raw_data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw dat files with correct delimiters, encodings, and column names.
    
    Args:
        raw_data_dir: Path to the extracted 'ml-1m' directory.
        
    Returns:
        Tuple of (ratings_df, users_df, movies_df).
    """
    ratings_path = raw_data_dir / "ratings.dat"
    users_path = raw_data_dir / "users.dat"
    movies_path = raw_data_dir / "movies.dat"
    
    logger.info("Loading ratings.dat...")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="ISO-8859-1",
        dtype={"user_id": int, "movie_id": int, "rating": int, "timestamp": int}
    )
    
    logger.info("Loading users.dat...")
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="ISO-8859-1",
        dtype={"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str}
    )
    
    logger.info("Loading movies.dat...")
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
        dtype={"movie_id": int, "title": str, "genres": str}
    )
    
    return ratings, users, movies

def preprocess_and_split(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    min_interactions: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filters users with few interactions, splits ratings temporally per user,
    and merges metadata to produce final train, val, and test splits.
    
    Args:
        ratings: Raw ratings DataFrame.
        users: Raw users DataFrame.
        movies: Raw movies DataFrame.
        min_interactions: Minimum interactions required per user.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # 1. Filter out users with < min_interactions
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()
    logger.info(f"Filtered out {len(user_counts) - len(valid_users)} users with < {min_interactions} interactions.")
    logger.info(f"Remaining interactions: {len(filtered_ratings)}")
    
    # 2. Sort interactions chronologically with movie_id as deterministic tie-breaker
    sorted_ratings = filtered_ratings.sort_values(
        by=["user_id", "timestamp", "movie_id"]
    ).reset_index(drop=True)
    
    # 3. Vectorized rank calculation from latest to oldest
    # interaction_index goes from 0 (oldest) to N-1 (newest) per user
    sorted_ratings["interaction_index"] = sorted_ratings.groupby("user_id").cumcount()
    sorted_ratings["total_interactions"] = sorted_ratings.groupby("user_id")["user_id"].transform("count")
    
    # reverse_rank: 0 is latest, 1 is second latest, >= 2 is older train interactions
    sorted_ratings["reverse_rank"] = sorted_ratings["total_interactions"] - 1 - sorted_ratings["interaction_index"]
    
    # Split indices
    test_mask = sorted_ratings["reverse_rank"] == 0
    val_mask = sorted_ratings["reverse_rank"] == 1
    train_mask = sorted_ratings["reverse_rank"] >= 2
    
    test_ratings = sorted_ratings[test_mask].drop(columns=["interaction_index", "total_interactions", "reverse_rank"])
    val_ratings = sorted_ratings[val_mask].drop(columns=["interaction_index", "total_interactions", "reverse_rank"])
    train_ratings = sorted_ratings[train_mask].drop(columns=["interaction_index", "total_interactions", "reverse_rank"])
    
    # 4. Merge metadata (left join to preserve all ratings)
    logger.info("Merging user and movie metadata with splits...")
    def enrich_split(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="movie_id", how="left")
        return df
        
    train_df = enrich_split(train_ratings)
    val_df = enrich_split(val_ratings)
    test_df = enrich_split(test_ratings)
    
    logger.info(f"Split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: str = "data/processed") -> None:
    """
    Saves processed splits as CSV files.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(out_path / "train.csv", index=False)
    val.to_csv(out_path / "val.csv", index=False)
    test.to_csv(out_path / "test.csv", index=False)
    logger.info(f"Saved processed datasets to {out_path}")

def main() -> None:
    raw_data_dir = Path("data/raw/ml-1m").resolve()
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found at {raw_data_dir}. Run download script first.")
        
    ratings, users, movies = load_raw_data(raw_data_dir)
    train, val, test = preprocess_and_split(ratings, users, movies)
    save_splits(train, val, test)

if __name__ == "__main__":
    main()
```

---

## 5. Data Leakage & Integrity Tests

### 5.1 Leakage Scenarios & Prevention
1. **User Overlap Invaliding Splits**: Since splits are partitioned per user, the same interaction (`user_id`, `movie_id`) must not appear in more than one split.
2. **Temporal Leakage**: A rating in `test.csv` must not have an earlier timestamp than any training or validation rating for that user. Similarly, a validation rating must not precede a training rating.
3. **Empty Splits**: We must assert that all output splits contain rows, and that every user has exactly one row in test, one row in validation, and at least three rows in train (since we require $\ge 5$ interactions).
4. **User Set Consistency**: The set of user IDs must be identical across train, validation, and test splits (since each user has at least 5 ratings, splitting leave-last-two-out preserves the user across all splits).

### 5.2 Unit Test Specification: `tests/test_data_pipeline.py`

This test suite uses synthetic data to verify user filtering, split logic, and leakage prevention without requiring the full MovieLens-1M dataset.

```python
import pytest
import pandas as pd
import numpy as np
from typing import Tuple
from src.data.preprocess import preprocess_and_split

@pytest.fixture
def mock_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic interaction data:
    - User 1: 5 interactions (Valid, should be split into 3 train, 1 val, 1 test)
    - User 2: 4 interactions (Invalid, should be completely filtered out)
    - User 3: 6 interactions (Valid, should be split into 4 train, 1 val, 1 test)
      Also tests multiple interactions within the same timestamp for User 3.
    """
    ratings_data = {
        "user_id": [
            1, 1, 1, 1, 1,          # User 1 (5 interactions)
            2, 2, 2, 2,             # User 2 (4 interactions, will be filtered)
            3, 3, 3, 3, 3, 3        # User 3 (6 interactions)
        ],
        "movie_id": [
            101, 102, 103, 104, 105,
            101, 102, 103, 104,
            201, 202, 203, 204, 205, 206
        ],
        "rating": [
            4, 5, 3, 2, 5,
            3, 3, 4, 2,
            5, 4, 3, 5, 2, 4
        ],
        "timestamp": [
            1000, 1001, 1002, 1003, 1004,  # User 1: sequential
            2000, 2001, 2002, 2003,        # User 2: sequential
            3000, 3001, 3002, 3003, 3003, 3003 # User 3: has tie-breaker timestamps
        ]
    }
    
    users_data = {
        "user_id": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "age": [25, 35, 18],
        "occupation": [4, 0, 15],
        "zip_code": ["94043", "02138", "10011"]
    }
    
    movies_data = {
        "movie_id": [101, 102, 103, 104, 105, 201, 202, 203, 204, 205, 206],
        "title": [f"Movie {i}" for i in range(11)],
        "genres": ["Action" for _ in range(11)]
    }
    
    return pd.DataFrame(ratings_data), pd.DataFrame(users_data), pd.DataFrame(movies_data)

def test_user_filtering(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies that users with < 5 interactions are filtered out.
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    all_users = set(train["user_id"]).union(val["user_id"]).union(test["user_id"])
    
    # User 2 should be filtered out
    assert 2 not in all_users
    # Users 1 and 3 should be kept
    assert 1 in all_users
    assert 3 in all_users

def test_split_size_invariants(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies sizes of test, val, and train splits per user.
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    # Every valid user must have exactly 1 validation interaction and exactly 1 test interaction
    for user_id in [1, 3]:
        user_train = train[train["user_id"] == user_id]
        user_val = val[val["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        
        assert len(user_val) == 1
        assert len(user_test) == 1
        
        # User 1: 5 interactions -> 3 train, 1 val, 1 test
        # User 3: 6 interactions -> 4 train, 1 val, 1 test
        expected_train_size = 3 if user_id == 1 else 4
        assert len(user_train) == expected_train_size

def test_user_sets_consistency(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies that the set of users in train, val, and test is identical.
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    train_users = set(train["user_id"])
    val_users = set(val["user_id"])
    test_users = set(test["user_id"])
    
    assert train_users == val_users
    assert val_users == test_users

def test_mutual_exclusivity(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies that the intersection of (user_id, movie_id) across splits is empty.
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    def get_interaction_pairs(df: pd.DataFrame) -> set:
        return set(zip(df["user_id"], df["movie_id"]))
        
    train_pairs = get_interaction_pairs(train)
    val_pairs = get_interaction_pairs(val)
    test_pairs = get_interaction_pairs(test)
    
    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)

def test_temporal_no_leakage(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies that no test or val interaction occurs earlier than any train interaction.
    Also verifies test is not earlier than val.
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    for user_id in [1, 3]:
        train_max_time = train[train["user_id"] == user_id]["timestamp"].max()
        val_time = val[val["user_id"] == user_id]["timestamp"].iloc[0]
        test_time = test[test["user_id"] == user_id]["timestamp"].iloc[0]
        
        # Test timestamp must be >= val timestamp
        assert test_time >= val_time
        # Val timestamp must be >= train max timestamp
        assert val_time >= train_max_time
        # Test timestamp must be >= train max timestamp
        assert test_time >= train_max_time

def test_deterministic_tie_breaker(mock_raw_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """
    Verifies that sorting handles identical timestamps deterministically using movie_id.
    User 3 has three interactions at timestamp 3003 (movie_ids 204, 205, 206).
    After sorting by (timestamp, movie_id):
    - 204: 4th interaction (goes to train)
    - 205: 5th interaction (goes to val)
    - 206: 6th interaction (goes to test)
    """
    ratings, users, movies = mock_raw_data
    train, val, test = preprocess_and_split(ratings, users, movies, min_interactions=5)
    
    user3_train = train[train["user_id"] == 3]
    user3_val = val[val["user_id"] == 3]
    user3_test = test[test["user_id"] == 3]
    
    # Verify that the test rating is indeed movie 206
    assert user3_test["movie_id"].iloc[0] == 206
    # Verify that the val rating is indeed movie 205
    assert user3_val["movie_id"].iloc[0] == 205
    # Verify that movie 204 is in train
    assert 204 in user3_train["movie_id"].values
```
