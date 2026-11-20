"""RP3beta and co-visitation as extra candidate sources.

The graph and the co-visitation matrix are built from the train slice only.
Alpha, beta and the BM25 parameters are chosen on 1000 validation users.
Test users are scored after that choice. The published CatBoost artifact is
the baseline; retrained models are written under reports/rp3_models and do
not replace artifacts/models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.models.train_ranker import _grouped_frame, _split_users, select_blend
from src.pipeline.recommend import RecommendationPipeline
from src.retrieval.candidates import unseen_candidates
from src.retrieval.graph_sources import (
    apply_popularity_beta,
    bm25_normalize,
    build_interaction_matrix,
    covisitation_weights,
    indices_to_item_ids,
    rp3_hop,
    score_users,
    top_indices,
)
from src.retrieval.index import FAISSIndex

REPORT_PATH = Path("reports/rp3_metrics.json")
MODEL_ROOT = Path("reports/rp3_models")
ALPHAS = (0.4, 0.6, 0.8, 1.0)
BETAS = (0.0, 0.3, 0.5, 0.7, 1.0)
TAU_DAYS = (7.0, 30.0, 90.0, 365.0)
K1_VALUES = (0.5, 1.2, 2.0)
B_VALUES = (0.25, 0.75)
SAMPLE_SIZE = 1000
SAMPLE_SEED = 42
POOL = 200
RECALL_KS = (50, 200, 1000)
DAY = 86400.0


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    context = load_context()
    report["split"] = context["split_info"]
    save_report(report)

    if "baseline" not in report:
        report["baseline"] = run_baseline(context)
        save_report(report)
    if "tuning" not in report:
        report["tuning"] = run_tuning(context)
        save_report(report)
    if "sources" not in report:
        report["sources"] = run_sources(context, report["tuning"])
        save_report(report)
    report.setdefault("ablations", {})
    for name in ("two_tower_retrained", "two_tower_rp3", "two_tower_covis", "all_three", "rp3_pure", "covis_pure"):
        if name in report["ablations"]:
            continue
        print(f"ablation {name}", flush=True)
        report["ablations"][name] = run_ablation(context, report, name)
        save_report(report)
    print(f"wrote {REPORT_PATH}", flush=True)


def load_context() -> dict:
    with open("configs/model_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    ratings = load_ratings(Path("data/raw/ml-1m"))
    train_df, val_df, test_df = temporal_split(ratings)
    assert_time_order(train_df, val_df, test_df)
    movies = load_movies(Path("data/raw/ml-1m"))
    users = load_users(Path("data/raw/ml-1m"))
    user2idx, item2idx = create_id_mappings(train_df)
    idx2item = np.empty(len(item2idx), dtype=np.int32)
    idx2user = np.empty(len(user2idx), dtype=np.int32)
    for item_id, index in item2idx.items():
        idx2item[index] = int(item_id)
    for user_id, index in user2idx.items():
        idx2user[index] = int(user_id)

    user_embs = np.load("artifacts/models/user_embeddings.npy")
    item_embs = np.load("artifacts/models/item_embeddings.npy")
    faiss_index = FAISSIndex.load(Path("artifacts/indexes/faiss_index.index"))
    if user_embs.shape[0] != len(user2idx) or item_embs.shape[0] != len(item2idx):
        raise RuntimeError("Embeddings are not aligned with the train id maps")
    if len(faiss_index) != len(item2idx):
        raise RuntimeError("FAISS index is not aligned with the train items")

    interactions, popularity = build_interaction_matrix(train_df, user2idx, item2idx)
    threshold = float(config["ranker"]["positive_threshold"])
    val_pos = positive_map(val_df, threshold)
    test_pos = positive_map(test_df, threshold)
    recent_k = int(config.get("two_tower", {}).get("recent_k", 10))
    split_info = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "users": int(len(user2idx)),
        "items": int(len(item2idx)),
        "interaction_edges": int(interactions.nnz),
        "val_positives": int(len(val_pos)),
        "test_positives": int(len(test_pos)),
        "test_positives_outside_train_catalog": int(
            sum(item not in item2idx for item in test_pos.values())
        ),
        "graph": "train interactions only; validation and test labels are not edges",
    }
    print(
        f"split users={split_info['users']} items={split_info['items']} "
        f"edges={split_info['interaction_edges']} test_positives={split_info['test_positives']}",
        flush=True,
    )
    return {
        "config": config,
        "train_df": train_df,
        "val_df": val_df,
        "movies": movies,
        "users": users,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "idx2user": idx2user,
        "user_embs": user_embs,
        "item_embs": item_embs,
        "faiss_index": faiss_index,
        "interactions": interactions,
        "popularity": popularity,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "recent_k": recent_k,
        "split_info": split_info,
        "user_features": build_user_features(train_df, movies, users),
        "item_features": build_item_features(train_df, movies),
        "genre_profiles": build_user_genre_profiles(train_df, movies),
        "recent_embs": build_recent_centroids(train_df, item_embs, user2idx, item2idx, recent_k),
    }


def run_baseline(context: dict) -> dict:
    test_users = sorted(context["test_pos"])
    ground_truth = {user_id: {item_id} for user_id, item_id in context["test_pos"].items()}
    production = production_ids(context, np.array([context["user2idx"][user] for user in test_users]), POOL)
    positives = np.array([context["test_pos"][user] for user in test_users], dtype=np.int32)
    production_recall = recall_at(production, positives, (50, 200))
    started = time.perf_counter()
    pipeline = make_pipeline(context)
    popularity = [int(row["item_id"]) for row in pipeline.recommend_cold_start(10)]
    recommendations = {}
    retrieval = {}
    for done, user_id in enumerate(test_users, start=1):
        recommendations[user_id] = [int(row["item_id"]) for row in pipeline.recommend(user_id, top_k=10)]
        history = pipeline.user_history.get(user_id, set())
        row = context["user2idx"][user_id]
        retrieval[user_id] = unseen_candidates(
            context["user_embs"][row : row + 1],
            context["faiss_index"],
            {int(index): int(item) for index, item in enumerate(context["idx2item"])},
            history,
            pipeline._retrieval_k(),
        )[:10]
        if done % 400 == 0:
            print(f"baseline recommend {done}/{len(test_users)}", flush=True)
    elapsed = time.perf_counter() - started
    full = evaluate_model(recommendations, ground_truth, 10)
    retrieval_metrics = evaluate_model(retrieval, ground_truth, 10)
    pop_recs = {user_id: popularity for user_id in ground_truth}
    print(
        f"baseline NDCG@10={full['ndcg@10']:.6f} Recall@10={full['recall@10']:.6f} "
        f"seconds={elapsed:.1f}",
        flush=True,
    )
    return {
        "n_users": len(test_users),
        "production_candidate_recall": production_recall,
        "full_pipeline": full,
        "retrieval_only": retrieval_metrics,
        "popularity": evaluate_model(pop_recs, ground_truth, 10),
        "recommend_seconds": elapsed,
        "recommend_seconds_per_user": elapsed / len(test_users),
    }


def run_tuning(context: dict) -> dict:
    sample_users = sample_validation_users(context["val_pos"])
    sample_rows = np.array([context["user2idx"][user] for user in sample_users], dtype=np.int32)
    positives = np.array([context["val_pos"][user] for user in sample_users], dtype=np.int32)
    rp3_rows, rp3_best = tune_rp3(context, sample_rows, positives)
    covis_rows, covis_best = tune_covis(context, sample_rows, positives)
    return {
        "sample_users": len(sample_users),
        "sample_seed": SAMPLE_SEED,
        "metric": "recall of the single validation item with rating >= 4",
        "rp3": {"grid": rp3_rows, "best": rp3_best},
        "covis": {"grid": covis_rows, "best": covis_best},
    }


def tune_rp3(context: dict, rows: np.ndarray, positives: np.ndarray) -> tuple[list[dict], dict]:
    trials = []
    best = None
    for alpha in ALPHAS:
        started = time.perf_counter()
        hop = rp3_hop(context["interactions"], context["popularity"], alpha)
        build_seconds = time.perf_counter() - started
        for beta in BETAS:
            similarity = apply_popularity_beta(hop, context["popularity"], beta)
            scored = score_users(context["interactions"], similarity, rows)
            item_ids = top_item_ids(scored, context["idx2item"], max(RECALL_KS))
            metrics = recall_at(item_ids, positives, RECALL_KS)
            trial = {
                "alpha": alpha,
                "beta": beta,
                "hop_seconds": build_seconds,
                **metrics,
            }
            trials.append(trial)
            print(
                f"rp3 alpha={alpha} beta={beta} recall@200={metrics['recall@200']:.4f}",
                flush=True,
            )
            if best is None or recall_key(trial) > recall_key(best):
                best = trial
    return trials, best


def tune_covis(context: dict, rows: np.ndarray, positives: np.ndarray) -> tuple[list[dict], dict]:
    trials = []
    best = None
    for tau_days in TAU_DAYS:
        started = time.perf_counter()
        weights = covisitation_weights(context["train_df"], context["item2idx"], tau_days * DAY)
        build_seconds = time.perf_counter() - started
        print(f"covis tau={tau_days} days built in {build_seconds:.1f}s", flush=True)
        for k1 in K1_VALUES:
            for b_value in B_VALUES:
                similarity = bm25_normalize(weights, context["popularity"], k1, b_value)
                scored = score_users(context["interactions"], similarity, rows)
                item_ids = top_item_ids(scored, context["idx2item"], max(RECALL_KS))
                metrics = recall_at(item_ids, positives, RECALL_KS)
                trial = {
                    "tau_days": tau_days,
                    "k1": k1,
                    "b": b_value,
                    "weight_seconds": build_seconds,
                    **metrics,
                }
                trials.append(trial)
                print(
                    f"covis tau={tau_days} k1={k1} b={b_value} recall@200={metrics['recall@200']:.4f}",
                    flush=True,
                )
                if best is None or recall_key(trial) > recall_key(best):
                    best = trial
    return trials, best


def run_sources(context: dict, tuning: dict) -> dict:
    packs, timings = build_packs(context, tuning)
    users = sorted(context["test_pos"])
    rows = np.array([context["user2idx"][user] for user in users], dtype=np.int32)
    positives = np.array([context["test_pos"][user] for user in users], dtype=np.int32)
    ranked = {
        name: packs[name]["top_ids"][rows]
        for name in ("two_tower", "rp3", "covis")
    }
    table = {name: recall_at(item_ids, positives, RECALL_KS) for name, item_ids in ranked.items()}
    unions = {
        "two_tower_rp3": ("two_tower", "rp3"),
        "two_tower_covis": ("two_tower", "covis"),
        "all_three": ("two_tower", "rp3", "covis"),
    }
    union_metrics = {}
    for name, members in unions.items():
        union_metrics[name] = union_recall([ranked[member] for member in members], positives, RECALL_KS)
    unique = {
        name: unique_share(ranked[name], [ranked[other] for other in ranked if other != name], RECALL_KS)
        for name in ranked
    }
    agreement = production_agreement(context, rows, ranked["two_tower"])
    return {
        "timings": timings,
        "recall": table,
        "union": union_metrics,
        "unique_fraction": unique,
        "two_tower_top200_matches_production_window": agreement,
    }


def run_ablation(context: dict, report: dict, name: str) -> dict:
    packs, _timings = build_packs(context, report["tuning"])
    if name == "rp3_pure":
        return pure_source(context, packs, "rp3")
    if name == "covis_pure":
        return pure_source(context, packs, "covis")
    members = {
        "two_tower_retrained": ("two_tower",),
        "two_tower_rp3": ("two_tower", "rp3"),
        "two_tower_covis": ("two_tower", "covis"),
        "all_three": ("two_tower", "rp3", "covis"),
    }[name]
    val_users = sorted(context["val_pos"])
    val_rows = np.array([context["user2idx"][user] for user in val_users], dtype=np.int32)
    train_frame = candidate_frame(context, packs, val_rows, members, context["val_pos"])
    hit_users = set(train_frame.loc[train_frame["label"] > 0, "user_id"].astype(int))
    train_frame = train_frame[train_frame["user_id"].isin(hit_users)].copy()
    print(f"{name} train users with the validation item in the pool: {len(hit_users)}", flush=True)
    ranker, mode, beta, holdout_ndcg = fit_ranker(train_frame, context["config"]["ranker"], name)
    test_users = sorted(context["test_pos"])
    test_rows = np.array([context["user2idx"][user] for user in test_users], dtype=np.int32)
    started = time.perf_counter()
    test_frame = candidate_frame(context, packs, test_rows, members, positives=None)
    recommendations = rerank(context, test_frame, ranker, mode, beta)
    elapsed = time.perf_counter() - started
    ground_truth = {user_id: {item_id} for user_id, item_id in context["test_pos"].items()}
    metrics = evaluate_model(recommendations, ground_truth, 10)
    print(
        f"{name} NDCG@10={metrics['ndcg@10']:.6f} Recall@10={metrics['recall@10']:.6f} "
        f"MAP@10={metrics['map@10']:.6f}",
        flush=True,
    )
    return {
        "sources": list(members),
        "train_users": len(hit_users),
        "blend_mode": mode,
        "blend_beta": beta,
        "holdout_ndcg@10": holdout_ndcg,
        "metrics": metrics,
        "infer_seconds": elapsed,
        "infer_seconds_per_user": elapsed / len(test_users),
    }


def pure_source(context: dict, packs: dict, source: str) -> dict:
    users = sorted(context["test_pos"])
    rows = np.array([context["user2idx"][user] for user in users], dtype=np.int32)
    recommendations = {}
    top = packs[source]["top_ids"]
    for user_id, row in zip(users, rows, strict=True):
        recommendations[user_id] = [int(item) for item in top[row, :10] if item >= 0]
    ground_truth = {user_id: {item_id} for user_id, item_id in context["test_pos"].items()}
    metrics = evaluate_model(recommendations, ground_truth, 10)
    print(
        f"{source} pure NDCG@10={metrics['ndcg@10']:.6f} Recall@10={metrics['recall@10']:.6f}",
        flush=True,
    )
    return {"sources": [source], "reranker": False, "metrics": metrics}


def build_packs(context: dict, tuning: dict) -> tuple[dict, dict]:
    cached = context.get("_source_cache")
    if cached is not None:
        return cached
    started = time.perf_counter()
    two_tower = two_tower_scores(context)
    two_tower_seconds = time.perf_counter() - started
    rp3_params = tuning["rp3"]["best"]
    started = time.perf_counter()
    hop = rp3_hop(context["interactions"], context["popularity"], rp3_params["alpha"])
    similarity = apply_popularity_beta(hop, context["popularity"], rp3_params["beta"])
    rp3_build = time.perf_counter() - started
    started = time.perf_counter()
    rp3_scores = score_users(context["interactions"], similarity, np.arange(len(context["user2idx"])))
    rp3_score_seconds = time.perf_counter() - started
    covis_params = tuning["covis"]["best"]
    started = time.perf_counter()
    weights = covisitation_weights(
        context["train_df"], context["item2idx"], covis_params["tau_days"] * DAY
    )
    covis_similarity = bm25_normalize(
        weights, context["popularity"], covis_params["k1"], covis_params["b"]
    )
    covis_build = time.perf_counter() - started
    started = time.perf_counter()
    covis_scores = score_users(
        context["interactions"], covis_similarity, np.arange(len(context["user2idx"]))
    )
    covis_score_seconds = time.perf_counter() - started
    k = max(RECALL_KS)
    packs = {
        "two_tower": pack_scores(two_tower, context["idx2item"], k),
        "rp3": pack_scores(rp3_scores, context["idx2item"], k),
        "covis": pack_scores(covis_scores, context["idx2item"], k),
    }
    pool_idx, pool_values = production_indices(context, POOL)
    packs["two_tower"]["pool_idx"] = pool_idx
    packs["two_tower"]["pool_values"] = pool_values
    n_users = len(context["user2idx"])
    timings = {
        "two_tower_search_seconds": two_tower_seconds,
        "two_tower_seconds_per_user": two_tower_seconds / n_users,
        "rp3_build_seconds": rp3_build,
        "rp3_score_seconds": rp3_score_seconds,
        "rp3_seconds_per_user": rp3_score_seconds / n_users,
        "covis_build_seconds": covis_build,
        "covis_score_seconds": covis_score_seconds,
        "covis_seconds_per_user": covis_score_seconds / n_users,
    }
    print(
        f"built sources rp3={rp3_build:.2f}s+{rp3_score_seconds:.2f}s "
        f"covis={covis_build:.2f}s+{covis_score_seconds:.2f}s",
        flush=True,
    )
    context["_source_cache"] = (packs, timings)
    return packs, timings


def candidate_frame(
    context: dict,
    packs: dict,
    rows: np.ndarray,
    members: tuple[str, ...],
    positives: dict | None,
) -> pd.DataFrame:
    user_chunks = []
    item_chunks = []
    columns: dict[str, list[np.ndarray]] = {}
    prefixes = {"two_tower": "tt", "rp3": "rp3", "covis": "covis"}
    source_features = len(members) > 1
    feature_names = []
    if source_features:
        for name in members:
            prefix = prefixes[name]
            feature_names.extend((f"{prefix}_score", f"{prefix}_rank", f"{prefix}_flag"))
    for name in feature_names:
        columns[name] = []
    retrieval_ranks = []
    for row in rows:
        chosen = []
        for name in members:
            indices, values = pool_slice(packs, name, row)
            chosen.append(indices[np.isfinite(values)])
        merged = np.unique(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int32)
        if merged.size == 0:
            continue
        user_chunks.append(np.full(merged.shape[0], context["idx2user"][row], dtype=np.int32))
        item_chunks.append(context["idx2item"][merged])
        retrieval_ranks.append(production_rank(packs, row, merged))
        if source_features:
            for name in members:
                prefix = prefixes[name]
                scores = packs[name]["scores"][row, merged]
                rank = packs[name]["rank"][row, merged]
                finite = np.isfinite(scores)
                in_pool = np.isin(merged, pool_slice(packs, name, row)[0])
                columns[f"{prefix}_score"].append(np.where(finite, scores, 0.0))
                columns[f"{prefix}_rank"].append(rank.astype(np.float64))
                columns[f"{prefix}_flag"].append(in_pool.astype(np.float64))
    frame = pd.DataFrame(
        {
            "user_id": np.concatenate(user_chunks),
            "item_id": np.concatenate(item_chunks),
            "retrieval_rank": np.concatenate(retrieval_ranks),
        }
    )
    for name, parts in columns.items():
        frame[name] = np.concatenate(parts)
    if positives is not None:
        mapped = frame["user_id"].map(positives)
        frame["label"] = (frame["item_id"] == mapped).fillna(False).astype(np.float64)
    features = build_pair_features(
        context["user_features"],
        context["item_features"],
        context["user_embs"],
        context["item_embs"],
        context["user2idx"],
        context["item2idx"],
        frame,
        context["genre_profiles"],
        context["recent_embs"],
    )
    return features


def fit_ranker(
    frame: pd.DataFrame,
    config: dict,
    name: str,
    save_dir: Path | None = None,
) -> tuple[CatBoostRanker, str, float, float]:
    labels = frame["label"].to_numpy(dtype=np.float64)
    train_users, holdout_users = _split_users(frame["user_id"].to_numpy(), config.get("seed", 42))
    train_part = frame[frame["user_id"].isin(train_users)].copy()
    holdout_part = frame[frame["user_id"].isin(holdout_users)].copy()
    train_part, y_train, qid_train = _grouped_frame(train_part, train_part["label"].to_numpy())
    holdout_part, y_holdout, qid_holdout = _grouped_frame(holdout_part, holdout_part["label"].to_numpy())
    del labels
    x_train, baseline_train = ranker_inputs(train_part)
    x_holdout, baseline_holdout = ranker_inputs(holdout_part)
    cat_features = [column for column in CAT_FEATURES if column in x_train.columns]
    text_features = [column for column in TEXT_FEATURES if column in x_train.columns]
    ranker = CatBoostRanker(config)
    ranker.fit(
        x_train,
        y_train,
        qid_train,
        x_holdout,
        y_holdout,
        qid_holdout,
        baseline_train=baseline_train,
        baseline_val=baseline_holdout,
        cat_features=cat_features,
        text_features=text_features,
    )
    mode, beta, ndcg = select_blend(holdout_part, ranker.predict(x_holdout), baseline_holdout)
    if save_dir is None:
        save_dir = MODEL_ROOT / name
    save_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(save_dir / "catboost_ranker.cbm")
    (save_dir / "ranker_blend.json").write_text(
        json.dumps({"mode": mode, "beta": beta, "holdout_ndcg@10": ndcg}, indent=2),
        encoding="utf-8",
    )
    return ranker, mode, beta, ndcg


def rerank(context: dict, frame: pd.DataFrame, ranker: CatBoostRanker, mode: str, beta: float) -> dict:
    model_frame, baseline = ranker_inputs(frame)
    if mode == "retrieval":
        scores = baseline
    else:
        scores = baseline + beta * ranker.predict(model_frame)
    scored = frame.loc[:, ["user_id", "item_id"]].copy()
    scored["score"] = scores
    recommendations = {}
    for user_id, group in scored.groupby("user_id", sort=False):
        ranked = group.sort_values("score", ascending=False, kind="mergesort")
        recommendations[int(user_id)] = ranked["item_id"].astype(int).head(10).tolist()
    popularity = popularity_ids(context)
    for user_id in context["test_pos"]:
        recommendations.setdefault(user_id, popularity)
    return recommendations


def two_tower_scores(context: dict) -> np.ndarray:
    queries = context["user_embs"].astype(np.float32, copy=False)
    distances, indices = context["faiss_index"].search(queries, len(context["item2idx"]))
    scores = np.full(distances.shape, -np.inf, dtype=np.float64)
    rows = np.arange(scores.shape[0])[:, None]
    scores[rows, indices] = distances
    seen = context["interactions"].tocoo()
    scores[seen.row, seen.col] = -np.inf
    return scores


def pack_scores(scores: np.ndarray, idx2item: np.ndarray, k: int) -> dict:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    rank = np.empty(order.shape, dtype=np.int32)
    rank[np.arange(scores.shape[0])[:, None], order] = np.arange(scores.shape[1], dtype=np.int32)
    top_idx = order[:, :k]
    top_values = np.take_along_axis(scores, top_idx, axis=1)
    return {
        "scores": scores,
        "rank": rank,
        "top_idx": top_idx,
        "top_values": top_values,
        "top_ids": indices_to_item_ids(top_idx, top_values, idx2item),
    }


def production_ids(context: dict, rows: np.ndarray, k: int) -> np.ndarray:
    """Same window as unseen_candidates: search 2k, keep k unseen."""
    queries = context["user_embs"][rows].astype(np.float32, copy=False)
    _distances, indices = context["faiss_index"].search(queries, k * 2)
    seen = context["interactions"][rows].toarray().astype(bool)
    output = np.full((len(rows), k), -1, dtype=np.int32)
    for position, raw in enumerate(indices):
        kept = 0
        for item_index in raw:
            item_index = int(item_index)
            if item_index < 0 or seen[position, item_index]:
                continue
            output[position, kept] = context["idx2item"][item_index]
            kept += 1
            if kept == k:
                break
    return output


def production_agreement(context: dict, rows: np.ndarray, full_ids: np.ndarray) -> dict:
    produced = production_ids(context, rows, POOL)
    full = full_ids[:, :POOL]
    same = np.all(produced == full, axis=1)
    return {
        "users": int(len(rows)),
        "identical_top200": int(same.sum()),
        "fraction": float(same.mean()),
    }


def pool_slice(packs: dict, name: str, row: int) -> tuple[np.ndarray, np.ndarray]:
    pack = packs[name]
    if "pool_idx" in pack:
        return pack["pool_idx"][row], pack["pool_values"][row]
    return pack["top_idx"][row, :POOL], pack["top_values"][row, :POOL]


def production_rank(packs: dict, row: int, item_indices: np.ndarray) -> np.ndarray:
    """FAISS position in the served top-200. Items outside that list get rank 200."""
    pool = packs["two_tower"].get("pool_idx")
    if pool is None:
        rank = packs["two_tower"]["rank"][row, item_indices]
        return np.minimum(rank, POOL).astype(np.float64)
    positions = {int(item): position for position, item in enumerate(pool[row]) if int(item) >= 0}
    return np.array([positions.get(int(item), POOL) for item in item_indices], dtype=np.float64)


def production_indices(context: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Item indices of the served two-tower list. Search window is 2k, same as the API."""
    queries = context["user_embs"].astype(np.float32, copy=False)
    distances, indices = context["faiss_index"].search(queries, k * 2)
    seen = context["interactions"].toarray().astype(bool)
    output = np.full((len(context["user2idx"]), k), -1, dtype=np.int32)
    values = np.full((len(context["user2idx"]), k), np.nan, dtype=np.float64)
    for position, raw in enumerate(indices):
        kept = 0
        for offset, item_index in enumerate(raw):
            item_index = int(item_index)
            if item_index < 0 or seen[position, item_index]:
                continue
            output[position, kept] = item_index
            values[position, kept] = float(distances[position, offset])
            kept += 1
            if kept == k:
                break
    return output, values


def top_item_ids(scores: np.ndarray, idx2item: np.ndarray, k: int) -> np.ndarray:
    values, indices = top_indices(scores, k)
    return indices_to_item_ids(indices, values, idx2item)


def recall_at(item_ids: np.ndarray, positives: np.ndarray, ks: tuple[int, ...]) -> dict:
    metrics = {"n": int(len(positives))}
    for k in ks:
        window = item_ids[:, : min(k, item_ids.shape[1])]
        hits = np.any(window == positives[:, None], axis=1)
        metrics[f"recall@{k}"] = float(hits.mean())
    return metrics


def union_recall(rankings: list[np.ndarray], positives: np.ndarray, ks: tuple[int, ...]) -> dict:
    metrics = {"n": int(len(positives))}
    for k in ks:
        hits = 0
        sizes = []
        for row, positive in enumerate(positives):
            pool = set()
            for ranking in rankings:
                pool.update(int(item) for item in ranking[row, :k] if int(item) >= 0)
            sizes.append(len(pool))
            hits += int(positive) in pool
        metrics[f"recall@{k}"] = hits / len(positives)
        metrics[f"mean_union_size@{k}"] = float(np.mean(sizes))
    return metrics


def unique_share(own: np.ndarray, others: list[np.ndarray], ks: tuple[int, ...]) -> dict:
    metrics = {}
    for k in ks:
        fractions = []
        for row in range(own.shape[0]):
            mine = [int(item) for item in own[row, :k] if int(item) >= 0]
            if not mine:
                continue
            rest = set()
            for other in others:
                rest.update(int(item) for item in other[row, :k] if int(item) >= 0)
            fractions.append(sum(item not in rest for item in mine) / len(mine))
        metrics[f"unique_fraction@{k}"] = float(np.mean(fractions))
    return metrics


def recall_key(trial: dict) -> tuple[float, float, float]:
    return (trial["recall@200"], trial["recall@50"], trial["recall@1000"])


def sample_validation_users(val_pos: dict) -> np.ndarray:
    users = np.array(sorted(val_pos), dtype=np.int32)
    generator = np.random.default_rng(SAMPLE_SEED)
    size = min(SAMPLE_SIZE, len(users))
    return np.sort(generator.choice(users, size=size, replace=False))


def positive_map(frame: pd.DataFrame, threshold: float) -> dict[int, int]:
    positive = frame[frame["rating"] >= threshold]
    return {int(row.user_id): int(row.item_id) for row in positive.itertuples(index=False)}


def assert_time_order(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_max = train_df.groupby("user_id")["timestamp"].max()
    val_ts = val_df.set_index("user_id")["timestamp"]
    test_ts = test_df.set_index("user_id")["timestamp"]
    aligned = pd.concat(
        [train_max.rename("train"), val_ts.rename("val"), test_ts.rename("test")],
        axis=1,
    ).dropna()
    if not bool((aligned["train"] <= aligned["val"]).all() and (aligned["val"] <= aligned["test"]).all()):
        raise RuntimeError("Per-user time order is not train <= validation <= test")


def make_pipeline(context: dict) -> RecommendationPipeline:
    idx2item = {int(index): int(item) for index, item in enumerate(context["idx2item"])}
    return RecommendationPipeline(
        user_embs=context["user_embs"],
        item_embs=context["item_embs"],
        faiss_index=context["faiss_index"],
        catboost_ranker=CatBoostRanker.load(Path("artifacts/models/catboost_ranker.cbm")),
        user_features=context["user_features"],
        item_features=context["item_features"],
        user2idx=context["user2idx"],
        item2idx=context["item2idx"],
        idx2item=idx2item,
        movies_df=context["movies"],
        ratings_df=context["train_df"],
        config=context["config"],
        train_df=context["train_df"],
        user_genre_profiles=context["genre_profiles"],
        user_recent_embs=context["recent_embs"],
    )


def popularity_ids(context: dict) -> list[int]:
    counts = context["train_df"]["item_id"].value_counts()
    return [int(item_id) for item_id in counts.head(10).index]


def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


if __name__ == "__main__":
    main()
