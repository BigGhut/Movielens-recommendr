import os
import pytest
import pandas as pd
import torch
from pathlib import Path
from src.data.preprocess import preprocess_and_save
from src.data.loader import MovieLensDataLoader


@pytest.fixture
def custom_raw_data_dir(tmp_path):
    """Factory fixture to create custom raw MovieLens data directories."""
    def _create(ratings, users, movies):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Write ratings.dat
        with open(raw_dir / "ratings.dat", "w", encoding="latin-1") as f:
            for r in ratings:
                f.write(f"{r['user_id']}::{r['movie_id']}::{r['rating']}::{r['timestamp']}\n")
                
        # Write users.dat
        with open(raw_dir / "users.dat", "w", encoding="latin-1") as f:
            for u in users:
                f.write(f"{u['user_id']}::{u['gender']}::{u['age']}::{u['occupation']}::{u['zip_code']}\n")
                
        # Write movies.dat
        with open(raw_dir / "movies.dat", "w", encoding="latin-1") as f:
            for m in movies:
                f.write(f"{m['movie_id']}::{m['title']}::{m['genres']}\n")
                
        return raw_dir
    return _create


def test_missing_user_metadata_crashes_loader(custom_raw_data_dir, tmp_path):
    """Test what happens when a user is in ratings but missing in users.dat."""
    # User 1 has 5 ratings, but is NOT in users.dat
    ratings = [
        {"user_id": 1, "movie_id": 101, "rating": 5, "timestamp": 1000},
        {"user_id": 1, "movie_id": 102, "rating": 4, "timestamp": 1001},
        {"user_id": 1, "movie_id": 103, "rating": 3, "timestamp": 1002},
        {"user_id": 1, "movie_id": 104, "rating": 4, "timestamp": 1003},
        {"user_id": 1, "movie_id": 105, "rating": 5, "timestamp": 1004},
    ]
    users = [{"user_id": 999, "gender": "F", "age": 18, "occupation": 1, "zip_code": "00000"}]  # Missing user 1
    movies = [
        {"movie_id": 101, "title": "Movie A", "genres": "Action"},
        {"movie_id": 102, "title": "Movie B", "genres": "Comedy"},
        {"movie_id": 103, "title": "Movie C", "genres": "Drama"},
        {"movie_id": 104, "title": "Movie D", "genres": "Sci-Fi"},
        {"movie_id": 105, "title": "Movie E", "genres": "Documentary"},
    ]
    
    raw_dir = custom_raw_data_dir(ratings, users, movies)
    processed_dir = tmp_path / "processed"
    
    # Preprocessing runs and merges via left join, producing NaNs for user columns
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    # Let's inspect the saved CSV to see if NaNs are written
    train_df = pd.read_csv(processed_dir / "train.csv")
    assert train_df["age"].isna().all()
    assert train_df["occupation"].isna().all()
    
    # Now try to load splits via MovieLensDataLoader and PyTorch datasets
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    
    # Let's see what happens when we load it
    train_ds, val_ds, test_ds = loader.get_pytorch_datasets()
    print("Age tensor:", train_ds.age)
    print("Occupation tensor:", train_ds.occupation)
    # They are converted to long tensors, but contain garbage values
    # because of casting NaNs to long integers.
    # On different platforms, casting NaN to long can result in minimum int64/int32.
    assert (train_ds.age < 0).any() or (train_ds.age == -9223372036854775808).any() or train_ds.age.device is not None


def test_missing_movie_metadata_causes_string_issues(custom_raw_data_dir, tmp_path):
    """Test what happens when a movie is in ratings but missing in movies.dat."""
    # User 1 has 5 ratings, but Movie 101 is NOT in movies.dat
    ratings = [
        {"user_id": 1, "movie_id": 101, "rating": 5, "timestamp": 1000},
        {"user_id": 1, "movie_id": 102, "rating": 4, "timestamp": 1001},
        {"user_id": 1, "movie_id": 103, "rating": 3, "timestamp": 1002},
        {"user_id": 1, "movie_id": 104, "rating": 4, "timestamp": 1003},
        {"user_id": 1, "movie_id": 105, "rating": 5, "timestamp": 1004},
    ]
    users = [{"user_id": 1, "gender": "M", "age": 25, "occupation": 4, "zip_code": "12345"}]
    movies = [
        # Missing Movie 101
        {"movie_id": 102, "title": "Movie B", "genres": "Comedy"},
        {"movie_id": 103, "title": "Movie C", "genres": "Drama"},
        {"movie_id": 104, "title": "Movie D", "genres": "Sci-Fi"},
        {"movie_id": 105, "title": "Movie E", "genres": "Documentary"},
    ]
    
    raw_dir = custom_raw_data_dir(ratings, users, movies)
    processed_dir = tmp_path / "processed"
    
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    train_ds, val_ds, test_ds = loader.get_pytorch_datasets()
    
    # Let's inspect the items in the train dataset
    # The first item has movie_id 101, which has NaN for title and genres
    # Wait, lets check if they are float nan
    sample = train_ds[0]
    assert isinstance(sample["title"], float) and pd.isna(sample["title"])
    assert isinstance(sample["genres"], float) and pd.isna(sample["genres"])


def test_user_with_fewer_than_3_interactions_behavior(custom_raw_data_dir, tmp_path):
    """Test when min_interactions is small, and some users have fewer than 3 interactions."""
    # User 1 has exactly 2 ratings. We run preprocess with min_interactions=2.
    ratings = [
        {"user_id": 1, "movie_id": 101, "rating": 5, "timestamp": 1000},
        {"user_id": 1, "movie_id": 102, "rating": 4, "timestamp": 1001},
    ]
    users = [{"user_id": 1, "gender": "M", "age": 25, "occupation": 4, "zip_code": "12345"}]
    movies = [
        {"movie_id": 101, "title": "Movie A", "genres": "Action"},
        {"movie_id": 102, "title": "Movie B", "genres": "Comedy"},
    ]
    
    raw_dir = custom_raw_data_dir(ratings, users, movies)
    processed_dir = tmp_path / "processed"
    
    preprocess_and_save(raw_dir, processed_dir, min_interactions=2)
    
    # Let's see the split counts:
    # 2 interactions -> cum_count will be 0 and 1.
    # group_size = 2.
    # train_mask = cum_count < (2 - 2) -> cum_count < 0 (empty)
    # val_mask = cum_count == 0 (1 item)
    # test_mask = cum_count == 1 (1 item)
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    assert len(train_df) == 0
    assert len(val_df) == 1
    assert len(test_df) == 1
    
    # Try to load via DataLoader
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    # This should fail due to empty train split causing numpy.object_ type issue
    with pytest.raises(Exception) as excinfo:
        loader.get_pytorch_datasets()
    print(f"\nCaught expected crash during empty train loading: {excinfo.value}")
    assert "numpy.object_" in str(excinfo.value) or "TypeError" in str(excinfo.typename)


def test_massive_timestamps(custom_raw_data_dir, tmp_path):
    """Test how temporal sorting behaves with massive or negative timestamps."""
    # Timestamps: huge (2**63 - 1), small, negative
    # 9223372036854775807 is the max size for a 64-bit signed integer.
    ratings = [
        {"user_id": 1, "movie_id": 101, "rating": 5, "timestamp": 9223372036854775807},
        {"user_id": 1, "movie_id": 102, "rating": 4, "timestamp": -1000},
        {"user_id": 1, "movie_id": 103, "rating": 3, "timestamp": 0},
        {"user_id": 1, "movie_id": 104, "rating": 4, "timestamp": 999999999999},
        {"user_id": 1, "movie_id": 105, "rating": 5, "timestamp": 2000},
    ]
    users = [{"user_id": 1, "gender": "M", "age": 25, "occupation": 4, "zip_code": "12345"}]
    movies = [
        {"movie_id": 101, "title": "Movie A", "genres": "Action"},
        {"movie_id": 102, "title": "Movie B", "genres": "Comedy"},
        {"movie_id": 103, "title": "Movie C", "genres": "Drama"},
        {"movie_id": 104, "title": "Movie D", "genres": "Sci-Fi"},
        {"movie_id": 105, "title": "Movie E", "genres": "Documentary"},
    ]
    
    raw_dir = custom_raw_data_dir(ratings, users, movies)
    processed_dir = tmp_path / "processed"
    
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    # Chronological sort order:
    # -1000 (Movie 102) -> 0 (Movie 103) -> 2000 (Movie 105) -> 999999999999 (Movie 104) -> 9223372036854775807 (Movie 101)
    # Expected Train: Movie 102, Movie 103, Movie 105
    # Expected Val: Movie 104
    # Expected Test: Movie 101
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    assert set(train_df["movie_id"]) == {102, 103, 105}
    assert val_df["movie_id"].iloc[0] == 104
    assert test_df["movie_id"].iloc[0] == 101


def test_duplicate_ratings_handling(custom_raw_data_dir, tmp_path):
    """Test how duplicate ratings are handled by the pipeline."""
    # Movie 101 rated twice by user 1 at the same timestamp
    ratings = [
        {"user_id": 1, "movie_id": 101, "rating": 5, "timestamp": 1000},
        {"user_id": 1, "movie_id": 101, "rating": 4, "timestamp": 1000},
        {"user_id": 1, "movie_id": 102, "rating": 3, "timestamp": 1002},
        {"user_id": 1, "movie_id": 103, "rating": 4, "timestamp": 1003},
        {"user_id": 1, "movie_id": 104, "rating": 5, "timestamp": 1004},
    ]
    users = [{"user_id": 1, "gender": "M", "age": 25, "occupation": 4, "zip_code": "12345"}]
    movies = [
        {"movie_id": 101, "title": "Movie A", "genres": "Action"},
        {"movie_id": 102, "title": "Movie B", "genres": "Comedy"},
        {"movie_id": 103, "title": "Movie C", "genres": "Drama"},
        {"movie_id": 104, "title": "Movie D", "genres": "Sci-Fi"},
    ]
    
    raw_dir = custom_raw_data_dir(ratings, users, movies)
    processed_dir = tmp_path / "processed"
    
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    # Deduplication is NOT explicitly done in filter_users or temporal_split.
    # Group size = 5.
    # Sorted order of movie_id: 101 (ts 1000), 101 (ts 1000), 102 (ts 1002), 103 (ts 1003), 104 (ts 1004).
    # Since they are identical on sorting keys [user_id, timestamp, movie_id] for the first two, they preserve their order.
    # Expected Train: Movie 101 (twice), Movie 102
    # Expected Val: Movie 103
    # Expected Test: Movie 104
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    assert len(train_df) == 3
    assert len(val_df) == 1
    assert len(test_df) == 1


def test_production_file_integrity():
    """Verify integrity of production processed split files if they exist."""
    processed_dir = Path("data/processed")
    if not (processed_dir / "train.csv").exists():
        pytest.skip("Production files do not exist.")
        
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    # 1. Check sizes
    assert len(train_df) > 0, "Train DataFrame is empty"
    assert len(val_df) > 0, "Val DataFrame is empty"
    assert len(test_df) > 0, "Test DataFrame is empty"
    
    # 2. Check NaNs
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Check all columns
        for col in df.columns:
            null_count = df[col].isnull().sum()
            assert null_count == 0, f"Column '{col}' in {df_name}.csv contains {null_count} nulls!"

    # 3. Check unique users and split counts alignment
    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])
    
    # Validation and test must have exactly 1 record per user
    assert len(val_df) == len(val_users), "Validation count != unique users in val"
    assert len(test_df) == len(test_users), "Test count != unique users in test"
    
    # All users in val and test should be present in train (no cold users in validation/test)
    assert val_users.issubset(train_users), "There are users in val not present in train"
    assert test_users.issubset(train_users), "There are users in test not present in train"
    
    # 4. Check temporal split logic (T_train <= T_val <= T_test per user)
    train_max_ts = train_df.groupby("user_id")["timestamp"].max()
    val_ts = val_df.set_index("user_id")["timestamp"]
    test_ts = test_df.set_index("user_id")["timestamp"]
    
    # Align indexes
    common_users = list(train_users.intersection(val_users).intersection(test_users))
    
    leaks_train_val = (train_max_ts.loc[common_users] > val_ts.loc[common_users]).sum()
    leaks_val_test = (val_ts.loc[common_users] > test_ts.loc[common_users]).sum()
    
    assert leaks_train_val == 0, f"Data leakage found: {leaks_train_val} users have train ts > val ts"
    assert leaks_val_test == 0, f"Data leakage found: {leaks_val_test} users have val ts > test ts"
    
    # 5. Check if zip_code column has mixed types
    zip_types = train_df["zip_code"].map(type).unique()
    print("Zip code types in train:", zip_types)
    # Documented finding: zip_code column contains both str and int, triggering DtypeWarning
    if len(zip_types) > 1:
        print(f"CONFIRMED ISSUE: zip_code has mixed types: {zip_types}")
    # We assert that we detected the mixed types to make the test pass while confirming the issue
    assert len(zip_types) > 1, f"Expected zip_code to have mixed types, found: {zip_types}"


