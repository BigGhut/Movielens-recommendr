from src.data.feature_store import (
    build_item_features,
    build_user_features,
    get_feature_names,
)


def test_feature_count():
    features = get_feature_names()
    assert len(features) >= 8

def test_user_features_no_nulls(synthetic_ratings, synthetic_movies, synthetic_users):
    user_feats = build_user_features(synthetic_ratings, synthetic_movies, synthetic_users)
    assert not user_feats.isnull().values.any()

def test_item_features_no_nulls(synthetic_ratings, synthetic_movies):
    item_feats = build_item_features(synthetic_ratings, synthetic_movies)
    assert not item_feats.isnull().values.any()
