import pytest
from pathlib import Path
import pandas as pd
import torch
from src.data.preprocess import (
    load_raw_data,
    filter_users,
    temporal_split,
    preprocess_and_save
)
from src.data.loader import MovieLensDataLoader, MovieLensDataset


@pytest.fixture
def base_mock_content():
    """Returns basic mock data strings to be customized by tests."""
    ratings = [
        "2::101::5::2000",
        "2::102::4::2001",
        "2::103::3::2002",
        "2::104::4::2003",
        "2::105::5::2004",
        "3::101::4::3000",
        "3::102::3::3001",
        "3::103::5::3002",
        "3::104::2::3003",
        "3::105::4::3004",
        "3::106::5::3005",
    ]
    users = [
        "2::M::56::16::70072",
        "3::M::25::15::55117",
    ]
    movies = [
        "101::Movie A::Action|Thriller",
        "102::Movie B::Comedy",
        "103::Movie C::Drama",
        "104::Movie D::Sci-Fi",
        "105::Movie E::Documentary",
        "106::Movie F::Horror",
    ]
    return ratings, users, movies


def write_mock_files(raw_dir: Path, ratings, users, movies):
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "ratings.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(ratings) + "\n")
    with open(raw_dir / "users.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(users) + "\n")
    with open(raw_dir / "movies.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(movies) + "\n")


def test_missing_raw_files_raised(tmp_path):
    """Test that missing raw directory/files raises FileNotFoundError."""
    non_existent = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        load_raw_data(non_existent)


def test_ratings_unseen_users_silent_corruption(tmp_path, base_mock_content):
    """Verify that users present in ratings but missing from users.dat cause silent data corruption.

    This tests referential integrity handling.
    """
    ratings, users, movies = base_mock_content
    # Add User 4 to ratings with 5 interactions, but do NOT add User 4 to users.dat
    ratings.extend([
        "4::101::5::4000",
        "4::102::4::4001",
        "4::103::3::4002",
        "4::104::4::4003",
        "4::105::5::4004",
    ])
    
    raw_dir = tmp_path / "raw"
    write_mock_files(raw_dir, ratings, users, movies)
    
    processed_dir = tmp_path / "processed"
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    # Files must exist
    assert (processed_dir / "train.csv").exists()
    
    # Try loading via loader
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    
    # PyTorch Dataset initialization does NOT fail, but silently corrupts age/occupation
    # values by converting NaN to the minimum int64 value.
    train_ds, _, _ = loader.get_pytorch_datasets()
    
    assert (train_ds.age < 0).any(), "Expected silent corruption of age tensor with negative values"
    assert (train_ds.occupation < 0).any(), "Expected silent corruption of occupation tensor with negative values"



def test_ratings_unseen_movies_nan_handling(tmp_path, base_mock_content):
    """Verify how the pipeline handles movies present in ratings but missing from movies.dat."""
    ratings, users, movies = base_mock_content
    # User 2 rates movie 999 which is not in movies.dat
    ratings[0] = "2::999::5::2000"
    
    raw_dir = tmp_path / "raw"
    write_mock_files(raw_dir, ratings, users, movies)
    
    processed_dir = tmp_path / "processed"
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    # PyTorch dataset handles lists of strings (title, genres)
    # If the join was a left join, missing title will be NaN (float).
    # Let's see if this compiles or fails.
    train_ds, _, _ = loader.get_pytorch_datasets()
    
    # If we access the first element, what is the type of title?
    sample = train_ds[0]
    # It might return float NaN, let's assert what it does.
    # Note: pandas read_csv reads empty/NaN strings as float nan.
    # This could cause string processing operations downstream to fail.
    assert pd.isna(sample["title"])


def test_single_rating_users_with_min_interactions_1(tmp_path, base_mock_content):
    """Verify behavior when min_interactions is set to 1 and we have a user with 1 rating.

    This tests split bounds and potential empty train/val sets.
    """
    ratings, users, movies = base_mock_content
    # Add User 4 with exactly 1 rating
    ratings.append("4::101::5::4000")
    users.append("4::M::25::4::48067")
    
    raw_dir = tmp_path / "raw"
    write_mock_files(raw_dir, ratings, users, movies)
    
    processed_dir = tmp_path / "processed"
    # Allow users with >= 1 interaction
    preprocess_and_save(raw_dir, processed_dir, min_interactions=1)
    
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    train, val, test = loader.load_splits()
    
    # User 4 has group size = 1.
    # Train/Val masks: train is empty, val is empty, test has the 1 rating.
    u4_train = train[train["user_id"] == 4]
    u4_val = val[val["user_id"] == 4]
    u4_test = test[test["user_id"] == 4]
    
    assert len(u4_train) == 0
    assert len(u4_val) == 0
    assert len(u4_test) == 1
    
    # Load PyTorch Datasets
    train_ds, val_ds, test_ds = loader.get_pytorch_datasets()
    # Check that datasets can be initialized even if train has missing users
    assert len(train_ds) > 0


def test_extreme_and_negative_timestamps(tmp_path, base_mock_content):
    """Verify that extremely large or negative timestamps are split and loaded correctly."""
    ratings, users, movies = base_mock_content
    # User 2 has timestamps:
    # 2000, 2001, 2002, 2003, 2004
    # Let's replace them with:
    # -1000 (negative timestamp)
    # 0
    # 2000
    # 9999999999 (fits in int64 but large)
    # 2**60 (massive int64 timestamp)
    ratings[0] = "2::101::5::-1000"
    ratings[1] = "2::102::4::0"
    ratings[2] = "2::103::3::2000"
    ratings[3] = "2::104::4::9999999999"
    ratings[4] = "2::105::5::1152921504606846976"  # 2**60
    
    raw_dir = tmp_path / "raw"
    write_mock_files(raw_dir, ratings, users, movies)
    
    processed_dir = tmp_path / "processed"
    preprocess_and_save(raw_dir, processed_dir, min_interactions=5)
    
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    train, val, test = loader.load_splits()
    
    # Verify order is kept correctly after preprocessing
    u2_train = train[train["user_id"] == 2]
    u2_val = val[val["user_id"] == 2]
    u2_test = test[test["user_id"] == 2]
    
    assert list(u2_train["timestamp"]) == [-1000, 0, 2000]
    assert u2_val["timestamp"].iloc[0] == 9999999999
    assert u2_test["timestamp"].iloc[0] == 1152921504606846976
    
    # Let's load the dataset and verify tensors
    train_ds, val_ds, test_ds = loader.get_pytorch_datasets()
    assert train_ds.timestamps[0] == -1000
    assert val_ds.timestamps[0] == 9999999999
    assert test_ds.timestamps[0] == 1152921504606846976
