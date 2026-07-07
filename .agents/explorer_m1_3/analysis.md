# MovieLens-1M Data Pipeline Investigation & Design (Milestone 1)

This report details the investigation findings and recommended implementation strategy for the MovieLens-1M Data Pipeline. It includes dataset downloading, loading, filtering, temporal splitting, and unit test designs to prevent data leakage.

## Executive Summary
This design implements a robust, reproducible, and leakage-free data preprocessing pipeline for the MovieLens-1M dataset. It details a secure download mechanism using Python's standard library, filters out sparse users, applies a strict per-user temporal split, and outlines a comprehensive unit test suite to guarantee data integrity.

---

## 1. Dataset Download & Loading Plan

### Download Strategy
The MovieLens-1M dataset is hosted by GroupLens at: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`.
To execute this download reliably on a user's machine without external dependencies, we use Python's built-in `urllib.request` and `zipfile` libraries. 

**Key Technical Details:**
1. **User-Agent Header**: GroupLens servers sometimes block requests that use python's default user-agent string (e.g., throwing a `HTTP Error 403: Forbidden`). To prevent this, our script explicitly sets a standard browser User-Agent header (e.g., `Mozilla/5.0`).
2. **Path Resolution**: The script creates two primary directories at the project root:
   - `data/raw/` for the zip archive and extracted contents.
   - `data/processed/` for the split datasets (`train.csv`, `validation.csv`, `test.csv`).
3. **Extraction & Clean-up**: After downloading the zip archive, it is extracted into `data/raw/` (which creates the directory `data/raw/ml-1m/`). The zip file is then deleted to conserve workspace storage.

### Loading Strategy
The MovieLens-1M files (`ratings.dat`, `users.dat`, and `movies.dat`) are text files using double-colons (`::`) as field delimiters and do not contain headers.
- **Encoding**: The files are encoded in `ISO-8859-1` (also known as `latin-1`). Using `utf-8` to read them will result in decoding crashes due to special characters in movie titles.
- **Engine**: Pandas `read_csv` requires `engine="python"` when delimiters of length greater than 1 (such as `::`) are used.
- **Column Schemas**:
  - `ratings.dat`: `user_id`, `movie_id`, `rating`, `timestamp`
  - `users.dat`: `user_id`, `gender`, `age`, `occupation`, `zip_code`
  - `movies.dat`: `movie_id`, `title`, `genres`

---

## 2. Filtering and Splitting Logic

### Filtering Logic
- **Constraint**: Filter out users with fewer than 5 ratings.
- **Rationale**: For each user, we extract exactly 1 rating for validation and 1 rating for testing. A minimum of 5 ratings guarantees that each valid user has at least 3 ratings left in the training set. If users with < 5 interactions were included, their training histories would be extremely sparse (e.g., 1 or 2 interactions) or non-existent, causing cold-start user training issues.
- **Implementation**:
  ```python
  user_counts = ratings["user_id"].value_counts()
  valid_users = user_counts[user_counts >= 5].index
  ratings_filtered = ratings[ratings["user_id"].isin(valid_users)].copy()
  ```

### Temporal Splitting Logic
To prevent **temporal data leakage** (using future interactions to predict past behavior), we split the ratings *per-user* based on interaction timestamps.
1. **Sort Interactions**: Sort the filtered ratings chronologically: first by `user_id`, then by `timestamp` ascending.
2. **Tie-Breaker**: To ensure a strictly deterministic split (crucial for reproducible offline evaluation), we sort secondary by `movie_id` ascending.
3. **Partitioning**:
   - For each user with sorted interactions $i_1, i_2, \dots, i_K$:
     - **Test**: The latest interaction $i_K$ (rank 0 from end).
     - **Validation**: The second-to-latest interaction $i_{K-1}$ (rank 1 from end).
     - **Train**: All earlier interactions $i_1, \dots, i_{K-2}$ (rank >= 2 from end).
4. **Vectorized Pandas Implementation** (avoiding slow python loops):
   ```python
   # Sort chronologically with a tie-breaker
   df = ratings_filtered.sort_values(by=["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
   
   # Add a reverse sequence rank per user (0 = latest, 1 = second latest, etc.)
   df["rank"] = df.groupby("user_id").cumcount(ascending=False)
   
   # Extract splits
   test_df = df[df["rank"] == 0].drop(columns=["rank"])
   val_df = df[df["rank"] == 1].drop(columns=["rank"])
   train_df = df[df["rank"] >= 2].drop(columns=["rank"])
   ```

### Metadata Merging
To support downstream feature extraction for the Re-ranking model (which requires at least 8 user, item, and cross features), the splits are joined with the `users` and `movies` metadata. This produces comprehensive processed tables in `data/processed/`.

---

## 3. Directory Structure & File Templates

### Proposed Directory Layout
We design the workspace layout as follows:
```
recsys-two-tower/
├── data/
│   ├── raw/
│   │   └── ml-1m/
│   │       ├── ratings.dat
│   │       ├── users.dat
│   │       └── movies.dat
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── users.csv
│       └── movies.csv
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── download.py
│       ├── preprocess.py
│       └── dataset.py
└── tests/
    └── test_data_pipeline.py
```

### File Templates

#### `src/data/download.py`
```python
import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_and_extract(data_dir: str = "data") -> None:
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    extract_path = os.path.join(raw_dir, "ml-1m")
    
    if os.path.exists(os.path.join(extract_path, "ratings.dat")):
        logger.info("MovieLens-1M files already exist in %s.", extract_path)
        return
        
    logger.info("Downloading MovieLens-1M dataset from %s...", MOVIELENS_1M_URL)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(MOVIELENS_1M_URL, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        logger.error("Failed to download MovieLens-1M dataset: %s", str(e))
        raise
        
    logger.info("Extracting zip archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
    except Exception as e:
        logger.error("Failed to extract zip file: %s", str(e))
        raise
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    logger.info("Dataset successfully downloaded and extracted to %s.", extract_path)

if __name__ == "__main__":
    download_and_extract()
```

#### `src/data/preprocess.py`
```python
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_ml_1m(raw_dir: str = "data/raw/ml-1m", processed_dir: str = "data/processed") -> None:
    os.makedirs(processed_dir, exist_ok=True)
    
    ratings_path = os.path.join(raw_dir, "ratings.dat")
    users_path = os.path.join(raw_dir, "users.dat")
    movies_path = os.path.join(raw_dir, "movies.dat")
    
    # Check if raw files exist
    for path in [ratings_path, users_path, movies_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required raw MovieLens file not found: {path}")
            
    logger.info("Loading MovieLens-1M raw files...")
    
    # Load raw data
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1"
    )
    
    users = pd.read_csv(
        users_path,
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1"
    )
    
    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )
    
    # 1. Filter out users with < 5 interactions
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    ratings_filtered = ratings[ratings["user_id"].isin(valid_users)].copy()
    
    logger.info(
        "Filtered dataset: %d users with >=5 ratings (%d total ratings kept, %d discarded).",
        len(valid_users),
        len(ratings_filtered),
        len(ratings) - len(ratings_filtered)
    )
    
    # 2. Sort chronologically (secondary by movie_id to ensure deterministic tie-breaking)
    ratings_filtered = ratings_filtered.sort_values(by=["user_id", "timestamp", "movie_id"]).reset_index(drop=True)
    
    # 3. User-level temporal splitting
    ratings_filtered["rank"] = ratings_filtered.groupby("user_id").cumcount(ascending=False)
    
    test_ratings = ratings_filtered[ratings_filtered["rank"] == 0].drop(columns=["rank"])
    val_ratings = ratings_filtered[ratings_filtered["rank"] == 1].drop(columns=["rank"])
    train_ratings = ratings_filtered[ratings_filtered["rank"] >= 2].drop(columns=["rank"])
    
    logger.info("Dataset sizes: Train=%d, Validation=%d, Test=%d", len(train_ratings), len(val_ratings), len(test_ratings))
    
    # 4. Enrich splits with user and movie metadata
    def enrich(df):
        df = df.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="movie_id", how="left")
        return df
        
    train_enriched = enrich(train_ratings)
    val_enriched = enrich(val_ratings)
    test_enriched = enrich(test_ratings)
    
    # Save files
    train_enriched.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_enriched.to_csv(os.path.join(processed_dir, "validation.csv"), index=False)
    test_enriched.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    # Save standard user and movie tables
    users.to_csv(os.path.join(processed_dir, "users.csv"), index=False)
    movies.to_csv(os.path.join(processed_dir, "movies.csv"), index=False)
    
    logger.info("Data preprocessing completed successfully. Files saved to %s.", processed_dir)

if __name__ == "__main__":
    preprocess_ml_1m()
```

#### `src/data/dataset.py`
```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MovieLensRetrievalDataset(Dataset):
    """
    Dataset wrapper for PyTorch Two-Tower Retrieval training.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
        # User characteristics
        self.user_ids = torch.tensor(self.df["user_id"].values, dtype=torch.long)
        self.genders = torch.tensor(self.df["gender"].map({"M": 0, "F": 1}).fillna(0).values, dtype=torch.long)
        self.ages = torch.tensor(self.df["age"].values, dtype=torch.long)
        self.occupations = torch.tensor(self.df["occupation"].values, dtype=torch.long)
        
        # Item characteristics
        self.movie_ids = torch.tensor(self.df["movie_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df["rating"].values, dtype=torch.float32)
        
        # Process genres into multi-hot array
        # MovieLens-1M contains 18 unique genres
        self.genre_list = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}
        self.genres_multi_hot = self._process_genres(self.df["genres"].values)
        
    def _process_genres(self, genres_series):
        multi_hots = np.zeros((len(genres_series), len(self.genre_list)), dtype=np.float32)
        for idx, genres_str in enumerate(genres_series):
            if pd.isna(genres_str):
                continue
            for genre in genres_str.split("|"):
                if genre in self.genre_to_idx:
                    multi_hots[idx, self.genre_to_idx[genre]] = 1.0
        return torch.tensor(multi_hots, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        return {
            "user_id": self.user_ids[idx],
            "gender": self.genders[idx],
            "age": self.ages[idx],
            "occupation": self.occupations[idx],
            "movie_id": self.movie_ids[idx],
            "rating": self.ratings[idx],
            "genres": self.genres_multi_hot[idx]
        }

def get_dataloader(csv_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = MovieLensRetrievalDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
```

---

## 4. Unit Test Specifications

We design a rigorous unit test suite `tests/test_data_pipeline.py` that verifies user filtering, checks for split completeness, and validates that there is **absolutely no temporal data leakage**.

### Code Specification for `tests/test_data_pipeline.py`
```python
import os
import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess_ml_1m

def test_data_pipeline_integrity(tmp_path):
    """
    Validates data loading, filtering, and split contracts on dummy data.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    
    # 1. Generate mock users.dat
    # Columns: UserID::Gender::Age::Occupation::Zip-code
    users_content = (
        "1::F::1::10::48067\n"
        "2::M::56::16::70072\n"
        "3::M::25::15::55117\n"
    )
    (raw_dir / "users.dat").write_text(users_content, encoding="latin-1")
    
    # 2. Generate mock movies.dat
    # Columns: MovieID::Title::Genres
    movies_content = (
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        "3::Grumpier Old Men (1995)::Comedy|Romance\n"
        "4::Waiting to Exhale (1995)::Comedy|Drama\n"
        "5::Father of the Bride Part II (1995)::Comedy\n"
        "6::Heat (1995)::Action|Crime|Thriller\n"
    )
    (raw_dir / "movies.dat").write_text(movies_content, encoding="latin-1")
    
    # 3. Generate mock ratings.dat
    # Columns: UserID::MovieID::Rating::Timestamp
    # - User 1 has 5 ratings (valid)
    # - User 2 has 4 ratings (invalid: should be filtered out)
    # - User 3 has 6 ratings (valid)
    ratings_content = (
        # User 1: 5 interactions
        "1::1::5::100\n"
        "1::2::4::200\n"
        "1::3::3::300\n"
        "1::4::4::400\n"  # Validation
        "1::5::5::500\n"  # Test
        # User 2: 4 interactions
        "2::1::3::100\n"
        "2::2::4::200\n"
        "2::3::5::300\n"
        "2::4::2::400\n"
        # User 3: 6 interactions
        "3::1::5::150\n"
        "3::2::5::250\n"
        "3::3::4::350\n"
        "3::4::3::450\n"
        "3::5::2::550\n"  # Validation
        "3::6::1::650\n"  # Test
    )
    (raw_dir / "ratings.dat").write_text(ratings_content, encoding="latin-1")
    
    # Run the preprocess script
    preprocess_ml_1m(str(raw_dir), str(processed_dir))
    
    # Load output datasets
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "validation.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    
    # --- TEST SUITE VERIFICATIONS ---
    
    # Assertion A: Verify User Filtering
    assert 2 not in train["user_id"].values, "User 2 has <5 ratings and should be filtered out from train."
    assert 2 not in val["user_id"].values, "User 2 has <5 ratings and should be filtered out from validation."
    assert 2 not in test["user_id"].values, "User 2 has <5 ratings and should be filtered out from test."
    
    # Assertion B: Non-Overlapping Splits
    # Check that individual interaction records are strictly separated between splits
    train_keys = set(zip(train["user_id"], train["movie_id"]))
    val_keys = set(zip(val["user_id"], val["movie_id"]))
    test_keys = set(zip(test["user_id"], test["movie_id"]))
    
    assert train_keys.isdisjoint(val_keys), "Overlapping interactions detected between Train and Validation!"
    assert train_keys.isdisjoint(test_keys), "Overlapping interactions detected between Train and Test!"
    assert val_keys.isdisjoint(test_keys), "Overlapping interactions detected between Validation and Test!"
    
    # Assertion C: Target Counts (Validation and Test have exactly 1 record per active user)
    active_users = {1, 3}
    assert set(train["user_id"]) == active_users
    assert set(val["user_id"]) == active_users
    assert set(test["user_id"]) == active_users
    
    assert len(val) == len(active_users)
    assert len(test) == len(active_users)
    
    # Assertion D: Zero Temporal Leakage
    for user in active_users:
        u_train = train[train["user_id"] == user]
        u_val = val[val["user_id"] == user]
        u_test = test[test["user_id"] == user]
        
        train_max_ts = u_train["timestamp"].max()
        val_ts = u_val["timestamp"].iloc[0]
        test_ts = u_test["timestamp"].iloc[0]
        
        # Test timestamp must be strictly greater than or equal to validation timestamp,
        # which must be strictly greater than or equal to the maximum training timestamp.
        assert val_ts >= train_max_ts, f"User {user} validation rating timestamp is earlier than training."
        assert test_ts >= val_ts, f"User {user} test rating timestamp is earlier than validation."
        
    # Assertion E: Check metadata joins
    # Train should have demographic information and titles
    for df in [train, val, test]:
        assert "gender" in df.columns
        assert "age" in df.columns
        assert "occupation" in df.columns
        assert "title" in df.columns
        assert "genres" in df.columns
        assert df["title"].isna().sum() == 0, "Movie titles should be fully merged and non-null."
```
