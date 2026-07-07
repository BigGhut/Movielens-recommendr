import argparse
from pathlib import Path
import numpy as np
import lightgbm as lgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--num-features", type=int, default=8)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock training data
    np.random.seed(42)
    X = np.random.randn(200, args.num_features)
    
    # Ensure every feature contributes to the target to force splits on all features
    y_raw = np.zeros(200)
    for i in range(args.num_features):
        y_raw += (X[:, i] > 0.0).astype(float) * (i + 1)
    
    median_val = np.median(y_raw)
    y = (y_raw >= median_val).astype(int)
    
    # Create LightGBM Dataset
    train_data = lgb.Dataset(X, label=y)
    
    # Train a booster with parameters that will split on all features
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.2,
        "min_data_in_leaf": 5,
        "verbose": -1
    }
    
    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=30
    )
    
    # Save the model
    gbm.save_model(str(model_dir / "reranker.lgb"))
    
    print(f"Mock GBDT training complete. Saved models/reranker.lgb in {model_dir}")

if __name__ == "__main__":
    main()
