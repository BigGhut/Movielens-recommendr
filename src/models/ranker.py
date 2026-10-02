from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker as CatBoostRankerModel

CAT_FEATURES = ['most_common_genre', 'gender', 'occupation', 'zip']
TEXT_FEATURES = ['genres']
# One FAISS position is this many baseline units. Smaller than 1 so a tree
# grown at the configured learning rate can move an item past its neighbor.
RANK_BASELINE_SCALE = 50.0


def finalize_ranker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Column prep shared by training and inference.

    Cosine similarity stays in the frame: the trees need it as a feature so they
    can correct retrieval instead of replacing it blindly. Ids are not features.
    """
    prepared = frame.copy()
    for col in CAT_FEATURES + TEXT_FEATURES:
        if col in prepared.columns:
            prepared[col] = prepared[col].fillna("Unknown").astype(str)
    for col in TEXT_FEATURES:
        if col in prepared.columns:
            prepared[col] = prepared[col].str.replace('|', ' ', regex=False)
    return prepared.drop(columns=['user_id', 'item_id', 'title', 'label'], errors='ignore')


def ranker_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Model matrix and the retrieval baseline.

    The baseline is the FAISS order (-rank). Trees only see the other features,
    so they learn a correction on top of retrieval instead of replacing it.
    """
    baseline = -frame['retrieval_rank'].to_numpy(dtype=np.float64) / RANK_BASELINE_SCALE
    features = finalize_ranker_frame(frame).drop(columns=['retrieval_rank'], errors='ignore')
    return features, baseline


class CatBoostRanker:
    def __init__(self, config: dict):
        kwargs = {
            'iterations': config.get('iterations', 500),
            'learning_rate': config.get('learning_rate', 0.05),
            'depth': config.get('depth', 6),
            'l2_leaf_reg': config.get('l2_leaf_reg', 3.0),
            'random_seed': config.get('seed', 42),
            'loss_function': 'YetiRank',
            'eval_metric': 'NDCG:top=10',
            'verbose': 50,
            'task_type': 'CPU'
        }
        if config.get('bootstrap_type'):
            kwargs['bootstrap_type'] = config['bootstrap_type']
        if config.get('subsample') is not None:
            kwargs['subsample'] = config['subsample']
        self.model = CatBoostRankerModel(**kwargs)
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, qid_train: np.ndarray,
            X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None, qid_val: np.ndarray | None = None,
            baseline_train: np.ndarray | None = None, baseline_val: np.ndarray | None = None,
            cat_features: list[str] | None = None, text_features: list[str] | None = None):
        """Обучает CatBoost-ранкер. Валидация необязательна."""
        from catboost import Pool
        train_kwargs = {
            "data": X_train,
            "label": y_train,
            "group_id": qid_train,
            "cat_features": cat_features,
            "text_features": text_features,
        }
        if baseline_train is not None:
            train_kwargs['baseline'] = baseline_train
        train_pool = Pool(**train_kwargs)
        fit_kwargs = {}
        if X_val is not None:
            val_kwargs = {
                "data": X_val,
                "label": y_val,
                "group_id": qid_val,
                "cat_features": cat_features,
                "text_features": text_features,
            }
            if baseline_val is not None:
                val_kwargs['baseline'] = baseline_val
            fit_kwargs['eval_set'] = Pool(**val_kwargs)
            fit_kwargs['early_stopping_rounds'] = 80
        self.model.fit(train_pool, **fit_kwargs)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает скоры для ранжирования."""
        return self.model.predict(X)
        
    def save(self, path: Path):
        """Сохраняет модель."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        
    @classmethod
    def load(cls, path: Path) -> 'CatBoostRanker':
        """Загружает модель."""
        instance = cls({})
        instance.model.load_model(str(path))
        return instance
        
    def feature_importance(self) -> pd.DataFrame:
        """Возвращает важность фич."""
        importance = self.model.get_feature_importance(type='PredictionValuesChange')
        names = self.model.feature_names_
        return pd.DataFrame({'feature': names, 'importance': importance}).sort_values('importance', ascending=False)
