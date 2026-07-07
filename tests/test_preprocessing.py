import pytest
from src.data.preprocessing import temporal_split

def test_temporal_split_no_leakage(synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, min_ratings=5)
    
    train_pairs = set(zip(train.user_id, train.item_id))
    test_pairs = set(zip(test.user_id, test.item_id))
    val_pairs = set(zip(val.user_id, val.item_id))
    
    assert len(train_pairs.intersection(test_pairs)) == 0
    assert len(train_pairs.intersection(val_pairs)) == 0
    assert len(val_pairs.intersection(test_pairs)) == 0

def test_temporal_split_order(synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, min_ratings=5)
    
    for u in test.user_id.unique():
        train_max_ts = train[train.user_id == u].timestamp.max()
        val_ts = val[val.user_id == u].timestamp.iloc[0]
        test_ts = test[test.user_id == u].timestamp.iloc[0]
        
        assert train_max_ts < val_ts
        assert val_ts < test_ts

def test_min_ratings_filter(synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, min_ratings=5)
    
    all_users = set(train.user_id).union(set(val.user_id)).union(set(test.user_id))
    assert 11 not in all_users # Юзер 11 имел только 3 рейтинга
    assert 1 in all_users # Юзер 1 имел 6 рейтингов

def test_splits_non_empty(synthetic_ratings):
    train, val, test = temporal_split(synthetic_ratings, min_ratings=5)
    assert not train.empty
    assert not val.empty
    assert not test.empty
