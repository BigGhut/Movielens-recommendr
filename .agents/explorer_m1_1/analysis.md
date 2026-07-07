# Analysis Report: MovieLens-1M Data Pipeline Design (Milestone 1)

This report details the investigation, design decisions, and proposed implementation strategy for the data pipeline of the MovieLens-1M Two-Tower Recommendation System.

---

## 1. Dataset Downloading and Loading Plan

### 1.1 Dataset Metadata
- **Name**: MovieLens-1M
- **Source**: GroupLens Research (University of Minnesota)
- **URL**: `http://files.grouplens.org/datasets/movielens/ml-1m.zip`
- **Size**: ~6 MB (zipped), ~24 MB (extracted)
- **Raw Files**:
  - `ratings.dat`: 1,000,209 ratings (UserID::MovieID::Rating::Timestamp)
  - `movies.dat`: 3,883 movies (MovieID::Title::Genres)
  - `users.dat`: 6,040 users (UserID::Gender::Age::Occupation::Zip-code)

### 1.2 Download Strategy
We propose a script `src/data/download.py` utilizing the Python standard library (`urllib.request` and `zipfile`) to download and extract the dataset. This keeps the download script dependency-free, avoiding the need for external libraries like `requests` or `wget`.
- **Liveness & Robustness**: The script will verify if raw files already exist locally before triggering a download, acting as a local cache check.
- **Cleanup**: It will remove the temporary zip archive after successful extraction to keep the storage clean.

### 1.3 Loading Strategy
Since the raw MovieLens-1M dataset uses `::` as a delimiter, loading it requires Pandas' Python parsing engine. Crucially, the movie data contains non-ASCII characters, necessitating the `ISO-8859-1` (or `latin-1`) encoding.
- **Parsing Parameters**:
  - `sep='::'`
  - `engine='python'`
  - `encoding='ISO-8859-1'`
  - `header=None` (manual column naming is required)

---

## 2. Filtering and Split Logic

### 2.1 Interaction Filtering
- **Requirement**: Remove users with fewer than 5 interactions.
- **Observation**: The raw MovieLens-1M dataset is already pre-filtered by GroupLens to include only users who have rated at least 20 movies. Thus, the $\ge 5$ threshold is mathematically satisfied for all users out-of-the-box.
- **Implementation**: Despite the dataset pre-filtering, we must write explicit filtering code to guarantee robustness, ensuring the pipeline will work correctly on sub-sampled datasets or future dataset variations.
- **Pandas Vectorization**:
  ```python
  user_counts = ratings["user_id"].value_counts()
  keep_users = user_counts[user_counts >= 5].index
  ratings_filtered = ratings[ratings["user_id"].isin(keep_users)].copy()
  ```

### 2.2 Temporal Split Logic
- **Requirement**: For each user, the latest interaction goes to `test`, the second-to-latest goes to `validation`, and all others go to `train`.
- **Ordering & Determinism**:
  - Timestamps in MovieLens-1M are represented in seconds. 
  - To make the split fully deterministic and reproducible, we sort interactions by `['user_id', 'timestamp', 'movie_id']`. The `movie_id` acts as a secondary sort key to break ties when multiple ratings share the same timestamp.
- **Vectorized Split Algorithm**:
  Using `groupby` and `cumcount` to rank interactions chronologically:
  1. Assign a row number `user_rank` (0 to $N-1$) for each user's sorted ratings.
  2. Compute total count `user_total` ($N$) for each user.
  3. Map splits based on rank:
     - **Test**: `user_rank == user_total - 1` (exactly 1 interaction per user)
     - **Validation**: `user_rank == user_total - 2` (exactly 1 interaction per user)
     - **Train**: `user_rank < user_total - 2` (remaining $N-2$ interactions)
- **Mathematical Safety**:
  Since $N \ge 5$ (after filtering), we guarantee that:
  - Every user has at least $5 - 2 = 3$ training samples.
  - Every user has exactly 1 validation sample.
  - Every user has exactly 1 test sample.
  This prevents the user cold-start evaluation problem, ensuring user embeddings are learned in training before they are evaluated.

---

## 3. Recommended Directory Structure and Code Templates

We recommend the following layout inside the project:

```
recsys-two-tower/
├── data/
│   ├── raw/
│   │   └── ml-1m/
│   │       ├── ratings.dat
│   │       ├── movies.dat
│   │       └── users.dat
│   └── processed/
│       ├── train.csv (or train.parquet)
│       ├── val.csv (or val.parquet)
│       └── test.csv (or test.parquet)
├── src/
│   └── data/
│       ├── __init__.py
│       ├── download.py       # Downloads & extracts zip file
│       ├── preprocess.py     # Filters, splits, and saves dataset
│       └── loader.py         # Loads train/val/test splits for downstream usage
└── tests/
    └── test_data.py          # Pytest suite verifying correctness & non-leakage
```

### 3.1 Code Template: `src/data/download.py`
```python
import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_dataset(url: str, dest_path: Path) -> None:
    """Download MovieLens-1M dataset zip file."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def extract_dataset(zip_path: Path, extract_to: Path) -> None:
    """Extract zip archive containing MovieLens-1M raw files."""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extraction completed successfully.")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

def main(data_dir: str) -> None:
    """Main execution block to check cache, download, and extract."""
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    zip_path = raw_dir / "ml-1m.zip"
    extracted_dir = raw_dir / "ml-1m"

    required_files = ["ratings.dat", "movies.dat", "users.dat"]
    files_exist = all((extracted_dir / f).exists() for f in required_files)

    if files_exist:
        logger.info("Raw MovieLens-1M files already exist. Skipping download.")
        return

    try:
        download_dataset(URL, zip_path)
        extract_dataset(zip_path, raw_dir)
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Cleaned up temporary zip file.")
    except Exception as e:
        logger.error(f"Failed to acquire dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens-1M dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory path.")
    args = parser.parse_args()
    main(args.data_dir)
```

### 3.2 Code Template: `src/data/preprocess.py`
```python
import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load users, movies, and ratings from the raw directory."""
    ml_dir = raw_dir / "ml-1m"
    
    users_path = ml_dir / "users.dat"
    movies_path = ml_dir / "movies.dat"
    ratings_path = ml_dir / "ratings.dat"
    
    user_cols = ["user_id", "gender", "age", "occupation", "zip_code"]
    movie_cols = ["movie_id", "title", "genres"]
    rating_cols = ["user_id", "movie_id", "rating", "timestamp"]
    
    logger.info("Reading raw MovieLens-1M files...")
    users = pd.read_csv(users_path, sep="::", header=None, names=user_cols, engine="python", encoding="ISO-8859-1")
    movies = pd.read_csv(movies_path, sep="::", header=None, names=movie_cols, engine="python", encoding="ISO-8859-1")
    ratings = pd.read_csv(ratings_path, sep="::", header=None, names=rating_cols, engine="python", encoding="ISO-8859-1")
    
    return users, movies, ratings

def preprocess_and_split(
    users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame, min_interactions: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter low-interaction users and partition data chronologically."""
    # Step 1: User filtering
    user_counts = ratings["user_id"].value_counts()
    keep_users = user_counts[user_counts >= min_interactions].index
    ratings_filtered = ratings[ratings["user_id"].isin(keep_users)].copy()
    logger.info(f"Filtered ratings. Before: {len(ratings)}, After: {len(ratings_filtered)}")
    
    # Step 2: Deterministic sorting
    ratings_sorted = ratings_filtered.sort_values(by=["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    
    # Step 3: Per-user ranking
    ratings_sorted["user_rank"] = ratings_sorted.groupby("user_id").cumcount()
    ratings_sorted["user_total"] = ratings_sorted.groupby("user_id")["user_id"].transform("count")
    
    # Step 4: Split partitioning
    train_mask = ratings_sorted["user_rank"] < (ratings_sorted["user_total"] - 2)
    val_mask = ratings_sorted["user_rank"] == (ratings_sorted["user_total"] - 2)
    test_mask = ratings_sorted["user_rank"] == (ratings_sorted["user_total"] - 1)
    
    train_df = ratings_sorted[train_mask].copy()
    val_df = ratings_sorted[val_mask].copy()
    test_df = ratings_sorted[test_mask].copy()
    
    # Drop ranking helpers
    for df in [train_df, val_df, test_df]:
        df.drop(columns=["user_rank", "user_total"], inplace=True)
        
    # Step 5: Merge features
    logger.info("Merging user and movie metadata with splits...")
    train_merged = train_df.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    val_merged = val_df.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    test_merged = test_df.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    
    return train_merged, val_merged, test_merged

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    """Save processed splits to the processed directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Recommend saving as CSV for alignment with contract, but Parquet is an option if desired.
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    logger.info(f"Preprocessed splits saved successfully to {output_dir}")

def main(data_dir: str) -> None:
    data_path = Path(data_dir)
    users, movies, ratings = load_raw_data(data_path / "raw")
    train, val, test = preprocess_and_split(users, movies, ratings)
    save_splits(train, val, test, data_path / "processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split MovieLens-1M dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory path.")
    args = parser.parse_args()
    main(args.data_dir)
```

### 3.3 Code Template: `src/data/loader.py`
```python
from pathlib import Path
import pandas as pd

class MovieLensDataLoader:
    """Utility class to load prepared MovieLens-1M dataset splits."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        
    def load_split(self, split: str) -> pd.DataFrame:
        """Load a single split DataFrame (train, val, or test)."""
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split name: {split}")
            
        file_path = self.processed_dir / f"{split}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}. Please execute download and preprocess steps first."
            )
        return pd.read_csv(file_path)
        
    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, val, and test splits."""
        return (
            self.load_split("train"),
            self.load_split("val"),
            self.load_split("test")
        )
```

---

## 4. Proposed Unit Test Specifications

We design a suite of pytest unit tests to check both split sizes and data leakage. 
These tests will be placed in `tests/test_data.py`.

### 4.1 Test Targets
1. **User Filtering Integrity**: Confirms that users with fewer than $5$ ratings are dropped from the dataset (using simulated edge-case inputs).
2. **Split Shape and Boundaries**: Assures that each user in the processed dataset has exactly $1$ validation record, $1$ test record, and all prior records in training.
3. **No Interaction Overlap**: Verifies that no specific ratings row (identified by `(user_id, movie_id)`) belongs to multiple splits.
4. **Strict Temporal Sequence (No Temporal Leakage)**: Verifies that for every user:
   - The validation interaction is not chronologically earlier than any training interaction.
   - The test interaction is not chronologically earlier than the validation interaction.

### 4.2 Test Implementation Template: `tests/test_data.py`
```python
import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess_and_split

@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate dummy users, movies, and ratings dataframes representing typical/edge cases."""
    # 3 dummy users:
    # User 1: 6 interactions (valid, should split: 4 train, 1 val, 1 test)
    # User 2: 4 interactions (invalid, should be filtered out entirely)
    # User 3: 5 interactions (valid, should split: 3 train, 1 val, 1 test)
    ratings_dict = {
        "user_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "movie_id": [101, 102, 103, 104, 105, 106, 101, 102, 103, 104, 101, 102, 103, 104, 105],
        # Timestamps for user 1: chronological
        # Timestamps for user 3: has a tie at the end to test tie-breaker
        "timestamp": [1000, 1010, 1020, 1030, 1040, 1050, 1000, 1010, 1020, 1030, 2000, 2010, 2020, 2030, 2030],
        "rating": [5, 4, 3, 5, 2, 4, 3, 2, 5, 4, 4, 5, 3, 5, 1]
    }
    
    users_dict = {
        "user_id": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "age": [25, 35, 18],
        "occupation": [4, 7, 10],
        "zip_code": ["90210", "10001", "94102"]
    }
    
    movies_dict = {
        "movie_id": [101, 102, 103, 104, 105, 106],
        "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F"],
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror"]
    }
    
    return pd.DataFrame(users_dict), pd.DataFrame(movies_dict), pd.DataFrame(ratings_dict)

def test_user_filtering(dummy_data):
    users, movies, ratings = dummy_data
    train, val, test = preprocess_and_split(users, movies, ratings, min_interactions=5)
    
    # User 2 has 4 interactions, should be removed
    assert 2 not in train["user_id"].unique()
    assert 2 not in val["user_id"].unique()
    assert 2 not in test["user_id"].unique()
    
    # User 1 and 3 have >= 5, should be kept
    assert set(train["user_id"].unique()) == {1, 3}
    assert set(val["user_id"].unique()) == {1, 3}
    assert set(test["user_id"].unique()) == {1, 3}

def test_split_cardinality(dummy_data):
    users, movies, ratings = dummy_data
    train, val, test = preprocess_and_split(users, movies, ratings, min_interactions=5)
    
    # Each valid user must have exactly 1 record in val and test
    assert len(val) == 2
    assert len(test) == 2
    
    # User 1 has 6 total ratings: 4 train, 1 val, 1 test
    assert len(train[train["user_id"] == 1]) == 4
    # User 3 has 5 total ratings: 3 train, 1 val, 1 test
    assert len(train[train["user_id"] == 3]) == 3

def test_no_overlap(dummy_data):
    users, movies, ratings = dummy_data
    train, val, test = preprocess_and_split(users, movies, ratings, min_interactions=5)
    
    # Create unique keys
    train_keys = set(zip(train["user_id"], train["movie_id"]))
    val_keys = set(zip(val["user_id"], val["movie_id"]))
    test_keys = set(zip(test["user_id"], test["movie_id"]))
    
    assert train_keys.isdisjoint(val_keys), "Overlap found between train and validation!"
    assert train_keys.isdisjoint(test_keys), "Overlap found between train and test!"
    assert val_keys.isdisjoint(test_keys), "Overlap found between validation and test!"

def test_no_temporal_leakage(dummy_data):
    users, movies, ratings = dummy_data
    train, val, test = preprocess_and_split(users, movies, ratings, min_interactions=5)
    
    # Group train by user to get max training timestamp
    train_max_ts = train.groupby("user_id")["timestamp"].max()
    
    # Get val and test timestamps per user
    val_ts = val.set_index("user_id")["timestamp"]
    test_ts = test.set_index("user_id")["timestamp"]
    
    # Verify chronological ordering
    for user_id in [1, 3]:
        max_train = train_max_ts.loc[user_id]
        v_ts = val_ts.loc[user_id]
        t_ts = test_ts.loc[user_id]
        
        # Validation rating is at or after max training rating
        assert v_ts >= max_train, f"Val timestamp {v_ts} before train {max_train} for user {user_id}"
        # Test rating is at or after validation rating
        assert t_ts >= v_ts, f"Test timestamp {t_ts} before val {v_ts} for user {user_id}"
        
        # Tie-breaker verification (for User 3 who has a tie in timestamp at 2030)
        # Sort values: D (timestamp 2030, movie 104) and E (timestamp 2030, movie 105)
        # Because movie 104 < 105, Val should be movie 104 and Test should be movie 105
        if user_id == 3:
            val_movie = val.loc[val["user_id"] == 3, "movie_id"].values[0]
            test_movie = test.loc[test["user_id"] == 3, "movie_id"].values[0]
            assert val_movie == 104, f"Expected validation movie to be 104, got {val_movie}"
            assert test_movie == 105, f"Expected test movie to be 105, got {test_movie}"
```

---

## 5. Potential Bottlenecks and Caveats

1. **Memory Efficiency**: 
   Since MovieLens-1M is relatively small (1M rows), holding the datasets in-memory via Pandas is perfectly fine and runs in less than 5 seconds. If the pipeline scales to MovieLens-20M or 25M, standard Pandas `groupby` and `cumcount` could create memory pressure. For larger datasets, switching to `dask`, `polars`, or database-level sorting (SQL) is recommended.
2. **Same-Timestamp Tie-breaking**:
   MovieLens timestamps have one-second resolution. A user may rate multiple items in the same second. Sorting by `['timestamp', 'movie_id']` enforces determinism but could result in arbitrary ordering of same-second interactions. If the model relies heavily on strict order (e.g. sequence-based RNN architectures), these same-second ratings should be carefully handled (e.g. treated as a single session).
3. **Parquet vs CSV**:
   Parquet format is highly recommended over CSV for the processed directory. Parquet preserves data types natively (saving lists of genres, integer types, user age categories), and has faster read/write latency. However, CSV is also planned to maintain alignment with simple script requirements.
