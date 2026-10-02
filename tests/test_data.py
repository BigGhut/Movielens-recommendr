from pathlib import Path

import pytest

from src.data.loader import MovieLensDataLoader
from src.data.preprocessing import (
    filter_users,
    load_raw_data,
    preprocess_and_save,
    split_ratings_by_time,
)


@pytest.fixture
def mock_raw_data(tmp_path) -> Path:
    """Create a temporary directory with mock raw MovieLens-1M data files.
    
    Data includes:
    - User 1: 4 ratings (should be filtered out because < 5 interactions)
    - User 2: 5 ratings (borderline case: should be kept, split into 3 train, 1 val, 1 test)
    - User 3: 6 ratings (should be kept, split into 4 train, 1 val, 1 test)
    - User 4: 5 ratings with identical timestamps (to test deterministic tie-breaking via movie_id)
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    # Write ratings.dat (UserID::MovieID::Rating::Timestamp)
    ratings_content = [
        # User 1: 4 ratings (should be filtered out)
        "1::101::4::1000",
        "1::102::3::1001",
        "1::103::5::1002",
        "1::104::2::1003",
        # User 2: 5 ratings
        "2::101::5::2000",
        "2::102::4::2001",
        "2::103::3::2002",
        "2::104::4::2003",
        "2::105::5::2004",
        # User 3: 6 ratings
        "3::101::4::3000",
        "3::102::3::3001",
        "3::103::5::3002",
        "3::104::2::3003",
        "3::105::4::3004",
        "3::106::5::3005",
        # User 4: 5 ratings with same timestamps (1000)
        "4::103::4::1000",
        "4::101::5::1000",
        "4::105::3::1000",
        "4::102::2::1000",
        "4::104::5::1000",
    ]
    with open(raw_dir / "ratings.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(ratings_content) + "\n")
        
    # Write users.dat (UserID::Gender::Age::Occupation::Zip-code)
    users_content = [
        "1::F::1::10::48067",
        "2::M::56::16::70072",
        "3::M::25::15::55117",
        "4::F::45::7::02460",
    ]
    with open(raw_dir / "users.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(users_content) + "\n")
        
    # Write movies.dat (MovieID::Title::Genres)
    movies_content = [
        "101::Movie A::Action|Thriller",
        "102::Movie B::Comedy",
        "103::Movie C::Drama",
        "104::Movie D::Sci-Fi",
        "105::Movie E::Documentary",
        "106::Movie F::Horror",
    ]
    with open(raw_dir / "movies.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(movies_content) + "\n")
        
    return raw_dir


def test_user_filtering(mock_raw_data):
    """Test T1_F1_1: Verify filtering of low-interaction users (< 5 ratings)."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    
    # User 1 has 4 ratings, others have >= 5
    filtered_ratings = filter_users(ratings, min_interactions=5)
    
    # Assert User 1 is completely excluded
    assert 1 not in filtered_ratings["user_id"].unique()
    
    # Assert Users 2, 3, 4 are present
    assert set(filtered_ratings["user_id"].unique()) == {2, 3, 4}
    
    # Assert exact rating counts
    assert len(filtered_ratings) == 5 + 6 + 5


def test_user_filtering_borderline(mock_raw_data):
    """Test T2_F1_1 & T2_F1_2: Verify split for borderline user with 5 ratings, and filtering for user with 4 ratings."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    
    # Filter with min_interactions=5 (User 1 has 4, User 2 has 5)
    filtered_ratings = filter_users(ratings, min_interactions=5)
    
    # User 1 (4 ratings) must be excluded
    assert 1 not in filtered_ratings["user_id"].unique()
    
    # User 2 (5 ratings) must be included
    assert 2 in filtered_ratings["user_id"].unique()
    
    # Split the filtered data
    train, val, test = split_ratings_by_time(filtered_ratings)
    
    # For User 2 (5 ratings): 3 in train, 1 in validation, 1 in test
    u2_train = train[train["user_id"] == 2]
    u2_val = val[val["user_id"] == 2]
    u2_test = test[test["user_id"] == 2]
    
    assert len(u2_train) == 3
    assert len(u2_val) == 1
    assert len(u2_test) == 1


def test_temporal_split_logic(mock_raw_data):
    r"""Test T1_F1_2: Verify train/val/test temporal sequence ($T_{train} \le T_{val} \le T_{test}$)."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    filtered = filter_users(ratings, min_interactions=5)
    train, val, test = split_ratings_by_time(filtered)
    
    # For each user, verify temporal boundaries
    for user_id in [2, 3]:
        u_train = train[train["user_id"] == user_id]
        u_val = val[val["user_id"] == user_id]
        u_test = test[test["user_id"] == user_id]
        
        max_train_ts = u_train["timestamp"].max()
        val_ts = u_val["timestamp"].iloc[0]
        test_ts = u_test["timestamp"].iloc[0]
        
        assert max_train_ts <= val_ts
        assert val_ts <= test_ts


def test_split_exclusivity(mock_raw_data):
    """Test T1_F1_3 & T1_F1_4: Verify no overlap between splits (disjoint sets of records)."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    filtered = filter_users(ratings, min_interactions=5)
    train, val, test = split_ratings_by_time(filtered)
    
    # Verify no overlap based on key columns (user_id, movie_id, timestamp)
    def to_tuple_set(df):
        return set(zip(df["user_id"], df["movie_id"], df["timestamp"]))
        
    train_set = to_tuple_set(train)
    val_set = to_tuple_set(val)
    test_set = to_tuple_set(test)
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_proportions(mock_raw_data):
    """Test T1_F1_5: Verify exact counts of validation and test (exactly 1 record per user)."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    filtered = filter_users(ratings, min_interactions=5)
    _train, val, test = split_ratings_by_time(filtered)
    
    unique_users = filtered["user_id"].unique()
    num_users = len(unique_users)
    
    assert len(val) == num_users
    assert len(test) == num_users
    
    # Verify every user has exactly one validation and test record
    assert (val.groupby("user_id").size() == 1).all()
    assert (test.groupby("user_id").size() == 1).all()


def test_identical_timestamps(mock_raw_data):
    """Test T2_F1_3: Verify split logic when timestamps are identical (deterministic sorting by movie_id)."""
    ratings, _users, _movies = load_raw_data(mock_raw_data)
    filtered = filter_users(ratings, min_interactions=5)
    
    # User 4 has 5 ratings all with timestamp=1000 and movie_ids: [103, 101, 105, 102, 104]
    # Sorted order of movie_id: 101, 102, 103, 104, 105
    train, val, test = split_ratings_by_time(filtered)
    
    u4_train = train[train["user_id"] == 4]
    u4_val = val[val["user_id"] == 4]
    u4_test = test[test["user_id"] == 4]
    
    # Assert shapes
    assert len(u4_train) == 3
    assert len(u4_val) == 1
    assert len(u4_test) == 1
    
    # Assert deterministic split
    # Test should get movie 105 (largest movie_id)
    assert u4_test["movie_id"].iloc[0] == 105
    
    # Val should get movie 104 (second largest movie_id)
    assert u4_val["movie_id"].iloc[0] == 104
    
    # Train should get movies 101, 102, 103
    assert set(u4_train["movie_id"]) == {101, 102, 103}


def test_empty_ratings_input(tmp_path):
    """Test T2_F1_4: Verify preprocessing behavior with empty raw file. Raises ValueError."""
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir()
    
    # Create empty files
    (raw_dir / "ratings.dat").write_text("", encoding="latin-1")
    (raw_dir / "users.dat").write_text("1::M::20::1::12345\n", encoding="latin-1")
    (raw_dir / "movies.dat").write_text("101::Movie A::Action\n", encoding="latin-1")
    
    with pytest.raises(ValueError, match="Raw data file is empty|Ratings DataFrame is empty"):
        load_raw_data(raw_dir)


def test_preprocess_and_save_workflow(mock_raw_data, tmp_path):
    """Verify end-to-end preprocessing, merging, saving, and loader integration."""
    processed_dir = tmp_path / "processed"
    
    # Run pipeline
    preprocess_and_save(mock_raw_data, processed_dir, min_interactions=5)
    
    # Files must exist
    assert (processed_dir / "train.csv").exists()
    assert (processed_dir / "val.csv").exists()
    assert (processed_dir / "test.csv").exists()
    
    # Test loader
    loader = MovieLensDataLoader(data_dir=str(processed_dir))
    train, val, test = loader.load_splits()
    
    # Verify metadata merged successfully
    for df in [train, val, test]:
        assert "gender" in df.columns
        assert "age" in df.columns
        assert "occupation" in df.columns
        assert "zip_code" in df.columns
        assert "title" in df.columns
        assert "genres" in df.columns
        
    # Verify values are correctly merged (e.g. user 2 gender is 'M', Movie 101 is 'Movie A')
    assert (train[train["user_id"] == 2]["gender"] == "M").all()
    assert (train[train["movie_id"] == 101]["title"] == "Movie A").all()
    
    # Test PyTorch datasets
    train_ds, val_ds, test_ds = loader.get_pytorch_datasets()
    assert len(train_ds) == len(train)
    assert len(val_ds) == len(val)
    assert len(test_ds) == len(test)
    
    # Verify item content
    sample = train_ds[0]
    assert "user_id" in sample
    assert "movie_id" in sample
    assert "rating" in sample
    assert "timestamp" in sample
    assert "gender" in sample
    assert "age" in sample
    assert "occupation" in sample
    assert "zip_code" in sample
    assert "title" in sample
    assert "genres" in sample
