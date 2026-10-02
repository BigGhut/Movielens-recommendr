import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.feature_store import (
    build_item_features,
    build_pair_features,
    build_recent_centroids,
    build_user_features,
    build_user_genre_profiles,
)
from src.data.preprocessing import (
    create_id_mappings,
    load_movies,
    load_ratings,
    load_users,
    temporal_split,
)
from src.evaluation.metrics import evaluate_model
from src.models.ranker import CAT_FEATURES, TEXT_FEATURES, CatBoostRanker, ranker_inputs
from src.retrieval.candidates import unseen_candidates
from src.retrieval.index import FAISSIndex

BLEND_BETAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def build_labeled_pairs(
    history_df: pd.DataFrame,
    label_df: pd.DataFrame,
    user_embs: np.ndarray,
    user2idx: dict,
    idx2item: dict,
    faiss_index: FAISSIndex,
    top_k: int = 200,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Candidates the server will rank, labeled by the next held-out interaction.

    Train-history items are filtered at serve time, so they are not targets.
    A user is kept only when that next relevant item is already inside the
    retrieved list: the ranker can only reorder what retrieval returned.
    """
    history = {
        int(user_id): {int(item_id) for item_id in item_ids}
        for user_id, item_ids in history_df.groupby("user_id")["item_id"]
    }
    positives = {}
    for row in label_df.itertuples(index=False):
        if float(row.rating) >= positive_threshold:
            positives[int(row.user_id)] = int(row.item_id)

    rows = []
    labels = []
    kept = 0
    skipped = 0
    for user_id, positive_item in positives.items():
        if user_id not in user2idx:
            skipped += 1
            continue
        user_idx = user2idx[user_id]
        candidates = unseen_candidates(
            user_embs[user_idx : user_idx + 1],
            faiss_index,
            idx2item,
            history.get(user_id, set()),
            top_k,
        )
        if positive_item not in candidates:
            skipped += 1
            continue
        kept += 1
        for rank, item_id in enumerate(candidates):
            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "retrieval_rank": float(rank),
            })
            labels.append(1.0 if item_id == positive_item else 0.0)

    print(f"Группы с позитивом в top-{top_k}: {kept}, пропущены: {skipped}")
    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "retrieval_rank"]), np.array([])
    return pd.DataFrame(rows), np.asarray(labels, dtype=np.float64)


def _split_users(user_ids: np.ndarray, seed: int, train_fraction: float = 0.8) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    users = np.array(sorted({int(user_id) for user_id in user_ids}))
    rng.shuffle(users)
    cut = int(len(users) * train_fraction)
    cut = min(max(cut, 1), len(users) - 1)
    return set(users[:cut].tolist()), set(users[cut:].tolist())


def _grouped_frame(frame: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    order = np.argsort(frame["user_id"].to_numpy(), kind="mergesort")
    ordered = frame.iloc[order].reset_index(drop=True)
    ordered_labels = labels[order]
    return ordered, ordered_labels, ordered["user_id"].to_numpy()


def _ndcg_for_scores(frame: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    scored = frame[["user_id", "item_id"]].copy()
    scored["score"] = scores
    scored["label"] = frame["label"].to_numpy()
    recommendations = {}
    ground_truth = {}
    for user_id, group in scored.groupby("user_id", sort=False):
        positives = {int(item_id) for item_id in group.loc[group["label"] > 0, "item_id"]}
        if not positives:
            continue
        ranked = group.sort_values("score", ascending=False)["item_id"].astype(int).tolist()
        recommendations[int(user_id)] = ranked
        ground_truth[int(user_id)] = positives
    return evaluate_model(recommendations, ground_truth, k)["ndcg@10"]


def select_blend(frame: pd.DataFrame, predictions: np.ndarray, baseline: np.ndarray) -> tuple[str, float, float]:
    """Shrink the residual so the mix beats, or falls back to, FAISS order.

    Score is `-retrieval_rank + beta * residual`. beta=0 is retrieval order.
    """
    options = [("retrieval", 0.0, _ndcg_for_scores(frame, baseline))]
    for beta in BLEND_BETAS:
        ndcg = _ndcg_for_scores(frame, baseline + beta * predictions)
        options.append(("blend", beta, ndcg))
    # A tie keeps pure retrieval: a residual that does not win on holdout is not shipped.
    mode, beta, ndcg = max(options, key=lambda item: (item[2], -item[1]))
    print(f"Масштаб остатка: std={float(np.std(predictions)):.4f}")
    print("Подбор смеси на отложенных пользователях:")
    for option_mode, option_beta, option_ndcg in options:
        label = "retrieval" if option_mode == "retrieval" else f"beta={option_beta}"
        print(f"  {label}: NDCG@10={option_ndcg:.4f}")
    print(f"Выбрано: mode={mode}, beta={beta}, NDCG@10={ndcg:.4f}")
    return mode, beta, ndcg


def train_ranker(
    config: dict,
    history_df: pd.DataFrame,
    label_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_embs: np.ndarray,
    item_embs: np.ndarray,
    user2idx: dict,
    item2idx: dict,
    idx2item: dict,
    faiss_index: FAISSIndex,
    user_genre_profiles: dict,
    save_dir: Path,
    top_k: int = 200,
    user_recent_embs: np.ndarray = None,
) -> CatBoostRanker:
    print("Сбор кандидатов для ранкера...")
    pairs, labels = build_labeled_pairs(
        history_df, label_df, user_embs, user2idx, idx2item, faiss_index,
        top_k=top_k, positive_threshold=config["positive_threshold"],
    )
    if pairs.empty:
        raise RuntimeError("Нет групп с позитивом в списке кандидатов")

    print("Сбор признаков пар...")
    features = build_pair_features(
        user_features, item_features, user_embs, item_embs,
        user2idx, item2idx, pairs, user_genre_profiles, user_recent_embs,
    )
    features["label"] = labels
    train_users, holdout_users = _split_users(features["user_id"].to_numpy(), config.get("seed", 42))
    train_frame = features[features["user_id"].isin(train_users)].copy()
    holdout_frame = features[features["user_id"].isin(holdout_users)].copy()
    train_frame, y_train, qid_train = _grouped_frame(train_frame, train_frame["label"].to_numpy())
    holdout_frame, y_holdout, qid_holdout = _grouped_frame(holdout_frame, holdout_frame["label"].to_numpy())
    print(f"Пользователи ранкера: train={len(train_users)}, holdout={len(holdout_users)}")

    X_train, baseline_train = ranker_inputs(train_frame)
    X_holdout, baseline_holdout = ranker_inputs(holdout_frame)
    present_cat = [col for col in CAT_FEATURES if col in X_train.columns]
    present_text = [col for col in TEXT_FEATURES if col in X_train.columns]

    print("Обучение CatBoost...")
    ranker = CatBoostRanker(config)
    ranker.fit(
        X_train, y_train, qid_train,
        X_holdout, y_holdout, qid_holdout,
        baseline_train=baseline_train, baseline_val=baseline_holdout,
        cat_features=present_cat, text_features=present_text,
    )
    holdout_pred = ranker.predict(X_holdout)
    mode, beta, ndcg = select_blend(holdout_frame, holdout_pred, baseline_holdout)
    print(ranker.feature_importance().head(8).to_string(index=False))

    save_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(save_dir / "catboost_ranker.cbm")
    blend_path = save_dir / "ranker_blend.json"
    blend_path.write_text(json.dumps({
        "mode": mode,
        "beta": beta,
        "holdout_ndcg@10": ndcg,
    }, indent=2), encoding="utf-8")
    print(f"Модель сохранена в {save_dir / 'catboost_ranker.cbm'}")
    print(f"Смесь сохранена в {blend_path}")
    return ranker


if __name__ == "__main__":
    with open("configs/model_config.yaml", "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    config = full_config["ranker"]
    top_k = int(full_config.get("pipeline", {}).get("retrieval_top_k", 200))

    print("Загрузка данных для CatBoost...")
    data_dir = Path("data/raw/ml-1m")
    ratings = load_ratings(data_dir)
    movies_df = load_movies(data_dir)
    users_df = load_users(data_dir)
    train_df, val_df, _test_df = temporal_split(ratings)
    user2idx, item2idx = create_id_mappings(train_df)
    idx2item = {v: k for k, v in item2idx.items()}

    print("Генерация признаков по train...")
    user_features = build_user_features(train_df, movies_df, users_df)
    item_features = build_item_features(train_df, movies_df)
    user_genre_profiles = build_user_genre_profiles(train_df, movies_df)

    user_embs = np.load("artifacts/models/user_embeddings.npy")
    item_embs = np.load("artifacts/models/item_embeddings.npy")
    faiss_index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
    recent_k = int(full_config.get("two_tower", {}).get("recent_k", 10))
    user_recent_embs = build_recent_centroids(train_df, item_embs, user2idx, item2idx, recent_k)

    train_ranker(
        config, train_df, val_df,
        user_features, item_features,
        user_embs, item_embs,
        user2idx, item2idx, idx2item,
        faiss_index, user_genre_profiles,
        Path("artifacts/models"),
        top_k=top_k,
        user_recent_embs=user_recent_embs,
    )
