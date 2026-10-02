import subprocess
import sys

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def run_preprocess_baseline(preprocess_script, data_dir, tmp_path_factory):
    """Run preprocessing on baseline mock raw data."""
    raw_dir = tmp_path_factory.mktemp("raw_baseline")
    
    # We do NOT generate raw files here; mock_preprocess.py will generate them automatically
    # when it runs if they don't exist.
    cmd = [
        sys.executable,
        preprocess_script,
        "--raw-dir", str(raw_dir),
        "--processed-dir", str(data_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Preprocess script failed: {res.stderr}"
    
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, val, test

# --- Tier 1: Feature Coverage (F1) ---

def test_user_filtering(run_preprocess_baseline):
    """T1_F1_1: Verify filtering of low-interaction users (< 5 ratings)."""
    train, val, test = run_preprocess_baseline
    # User 3 has 4 ratings in mock data. Should be completely excluded.
    all_users = set(train["user_id"]).union(val["user_id"]).union(test["user_id"])
    assert 3 not in all_users, "User 3 (< 5 ratings) was not filtered out"
    # Users with >= 5 ratings (1, 2, 4, 5) should be present
    assert {1, 2, 4, 5}.issubset(all_users)

def test_temporal_split_logic(run_preprocess_baseline):
    """T1_F1_2: Verify train/val/test temporal sequence per user."""
    train, val, test = run_preprocess_baseline
    # For each user, T_train <= T_val <= T_test
    for user_id in set(train["user_id"]):
        user_train = train[train["user_id"] == user_id]
        user_val = val[val["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        
        t_train_max = user_train["timestamp"].max()
        t_val = user_val["timestamp"].iloc[0]
        t_test = user_test["timestamp"].iloc[0]
        
        assert t_train_max <= t_val, f"User {user_id}: Train max timestamp {t_train_max} > Val timestamp {t_val}"
        assert t_val <= t_test, f"User {user_id}: Val timestamp {t_val} > Test timestamp {t_test}"

def test_split_exclusivity(run_preprocess_baseline):
    """T1_F1_3: Verify no interaction overlap between splits."""
    train, val, test = run_preprocess_baseline
    
    def get_interaction_keys(df):
        return set(zip(df["user_id"], df["movie_id"], df["timestamp"]))
        
    train_keys = get_interaction_keys(train)
    val_keys = get_interaction_keys(val)
    test_keys = get_interaction_keys(test)
    
    assert train_keys.isdisjoint(val_keys), "Overlap found between Train and Val"
    assert train_keys.isdisjoint(test_keys), "Overlap found between Train and Test"
    assert val_keys.isdisjoint(test_keys), "Overlap found between Val and Test"

def test_data_leakage_absence(run_preprocess_baseline):
    """T1_F1_4: Ensure no test interactions are in train set."""
    train, _val, test = run_preprocess_baseline
    # No timestamp in test split is earlier than train split (per user and globally)
    # The global check requires that the test timestamps are after the train timestamps
    for user_id in set(train["user_id"]):
        user_train = train[train["user_id"] == user_id]
        user_test = test[test["user_id"] == user_id]
        
        assert (user_test["timestamp"].min() >= user_train["timestamp"].max()), f"User {user_id} test timestamp earlier than train"

def test_split_proportions(run_preprocess_baseline):
    """T1_F1_5: Verify exact counts of validation and test."""
    train, val, test = run_preprocess_baseline
    unique_users = set(train["user_id"])
    
    # Count of val/test records equals number of unique users
    assert len(val) == len(unique_users), f"Validation count {len(val)} != unique users {len(unique_users)}"
    assert len(test) == len(unique_users), f"Test count {len(test)} != unique users {len(unique_users)}"

# --- Tier 2: Boundary & Corner Cases (F1) ---

def test_user_exactly_5_ratings(run_preprocess_baseline):
    """T2_F1_1: Verify split for borderline user with 5 ratings."""
    train, val, test = run_preprocess_baseline
    # User 2 has exactly 5 ratings in mock raw.
    # Should have 3 in train, 1 in val, 1 in test.
    user_train_cnt = len(train[train["user_id"] == 2])
    user_val_cnt = len(val[val["user_id"] == 2])
    user_test_cnt = len(test[test["user_id"] == 2])
    
    assert user_train_cnt == 3
    assert user_val_cnt == 1
    assert user_test_cnt == 1

def test_user_exactly_4_ratings(run_preprocess_baseline):
    """T2_F1_2: Verify filtering for borderline user with 4 ratings."""
    train, val, test = run_preprocess_baseline
    # User 3 has exactly 4 ratings in mock raw.
    # User should be completely removed from all splits.
    assert len(train[train["user_id"] == 3]) == 0
    assert len(val[val["user_id"] == 3]) == 0
    assert len(test[test["user_id"] == 3]) == 0

def test_identical_timestamps(run_preprocess_baseline):
    """T2_F1_3: Verify split logic when timestamps are identical."""
    train, val, test = run_preprocess_baseline
    # User 5 has 5 ratings with identical timestamps.
    # Check that they were split deterministically (3 train, 1 val, 1 test).
    user_train_cnt = len(train[train["user_id"] == 5])
    user_val_cnt = len(val[val["user_id"] == 5])
    user_test_cnt = len(test[test["user_id"] == 5])
    
    assert user_train_cnt == 3
    assert user_val_cnt == 1
    assert user_test_cnt == 1

def test_empty_ratings_input(preprocess_script, tmp_path):
    """T2_F1_4: Verify preprocessing behavior with empty raw files."""
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir()
    
    # Create empty ratings.dat, users.dat, movies.dat
    (raw_dir / "ratings.dat").write_text("", encoding="latin-1")
    (raw_dir / "users.dat").write_text("", encoding="latin-1")
    (raw_dir / "movies.dat").write_text("", encoding="latin-1")
    
    processed_dir = tmp_path / "processed"
    
    cmd = [
        sys.executable,
        preprocess_script,
        "--raw-dir", str(raw_dir),
        "--processed-dir", str(processed_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # The script should exit with non-zero because of ValueError
    assert res.returncode != 0
    assert "ValueError" in res.stderr or "ValueError" in res.stdout

def test_extreme_temporal_range(preprocess_script, tmp_path):
    """T2_F1_5: Verify correctness when temporal split range is small."""
    raw_dir = tmp_path / "narrow_raw"
    raw_dir.mkdir()
    
    # 5 ratings for User 1 spanning exactly 1 second (same day)
    ratings = [
        "1::101::5::1000",
        "1::102::4::1000",
        "1::103::3::1000",
        "1::104::4::1001",
        "1::105::5::1001"
    ]
    users = ["1::M::25::4::48067"]
    movies = [
        "101::Movie A::Action",
        "102::Movie B::Action",
        "103::Movie C::Action",
        "104::Movie D::Action",
        "105::Movie E::Action"
    ]
    
    (raw_dir / "ratings.dat").write_text("\n".join(ratings) + "\n", encoding="latin-1")
    (raw_dir / "users.dat").write_text("\n".join(users) + "\n", encoding="latin-1")
    (raw_dir / "movies.dat").write_text("\n".join(movies) + "\n", encoding="latin-1")
    
    processed_dir = tmp_path / "processed"
    
    cmd = [
        sys.executable,
        preprocess_script,
        "--raw-dir", str(raw_dir),
        "--processed-dir", str(processed_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1
