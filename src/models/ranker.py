from catboost import CatBoostRanker as CatBoostRankerModel
import pandas as pd
import numpy as np
from pathlib import Path

class CatBoostRanker:
    def __init__(self, config: dict):
        kwargs = {
            'iterations': config.get('iterations', 500),
            'learning_rate': config.get('learning_rate', 0.05),
            'depth': config.get('depth', 6),
            'l2_leaf_reg': config.get('l2_leaf_reg', 3.0),
            'random_seed': config.get('seed', 42),
            'loss_function': 'YetiRank',
            'verbose': 50,
            'task_type': 'CPU'
        }
        self.model = CatBoostRankerModel(**kwargs)
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, qid_train: np.ndarray, 
            X_val: pd.DataFrame, y_val: np.ndarray, qid_val: np.ndarray, 
            baseline_train: np.ndarray = None, baseline_val: np.ndarray = None,
            cat_features: list[str] = None, text_features: list[str] = None):
        """Обучает CatBoost-ранкер."""
        from catboost import Pool
        train_pool = Pool(data=X_train, label=y_train, group_id=qid_train, cat_features=cat_features, text_features=text_features, baseline=baseline_train)
        val_pool = Pool(data=X_val, label=y_val, group_id=qid_val, cat_features=cat_features, text_features=text_features, baseline=baseline_val)
        
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50
        )
        
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
        importance = self.model.get_feature_importance()
        names = self.model.feature_names_
        return pd.DataFrame({'feature': names, 'importance': importance}).sort_values('importance', ascending=False)
