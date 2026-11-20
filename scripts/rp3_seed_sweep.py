"""Repeat the two rankers over five CatBoost seeds.

The candidate lists stay fixed. Only the CatBoost seed changes, and with it
the 80/20 holdout used for early stopping and the blend weight. The published
artifact is not retrained: its per-user NDCG is measured again with recommend.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rp3_bench as bench  # noqa: E402

from src.evaluation.metrics import evaluate_model, ndcg_at_k  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
REPORT = Path("reports/rp3_seed_sweep.json")
PUBLISHED_NDCG = 0.08665906317519945


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    tuning = json.loads(Path("reports/rp3_metrics.json").read_text(encoding="utf-8"))["tuning"]
    context = bench.load_context()
    users = sorted(context["test_pos"])
    positives = {user_id: context["test_pos"][user_id] for user_id in users}
    ground_truth = {user_id: {item_id} for user_id, item_id in positives.items()}

    if "published" not in report:
        print("published recommend", flush=True)
        report["published"] = published_per_user(context, users, positives, ground_truth)
        save(report)

    packs, _timings = bench.build_packs(context, tuning)
    frames = {}
    for name, members in (
        ("two_tower", ("two_tower",)),
        ("two_tower_rp3", ("two_tower", "rp3")),
    ):
        frames[name] = {
            "train": labeled_frame(context, packs, members),
            "test": bench.candidate_frame(
                context,
                packs,
                np.array([context["user2idx"][user] for user in users], dtype=np.int32),
                members,
                positives=None,
            ),
        }
        print(f"{name} train rows {len(frames[name]['train'])}", flush=True)

    report.setdefault("seeds", {})
    for seed in SEEDS:
        if str(seed) in report["seeds"]:
            continue
        print(f"seed {seed}", flush=True)
        report["seeds"][str(seed)] = run_seed(context, frames, users, positives, ground_truth, seed)
        save(report)
    report["summary"] = summarize(report, users)
    save(report)
    print(json.dumps(report["summary"], indent=2), flush=True)


def labeled_frame(context: dict, packs: dict, members: tuple[str, ...]):
    val_users = sorted(context["val_pos"])
    rows = np.array([context["user2idx"][user] for user in val_users], dtype=np.int32)
    frame = bench.candidate_frame(context, packs, rows, members, context["val_pos"])
    hit_users = set(frame.loc[frame["label"] > 0, "user_id"].astype(int))
    return frame[frame["user_id"].isin(hit_users)].copy()


def run_seed(context, frames, users, positives, ground_truth, seed: int) -> dict:
    config = dict(context["config"]["ranker"])
    config["seed"] = int(seed)
    result = {"seed": int(seed)}
    for name in ("two_tower", "two_tower_rp3"):
        started = time.perf_counter()
        ranker, mode, beta, holdout = bench.fit_ranker(
            frames[name]["train"],
            config,
            name,
            save_dir=Path("reports/rp3_seeds") / f"seed_{seed}" / name,
        )
        recommendations = bench.rerank(context, frames[name]["test"], ranker, mode, beta)
        metrics = evaluate_model(recommendations, ground_truth, 10)
        values = per_user(recommendations, users, positives)
        result[name] = {
            "metrics": metrics,
            "blend_mode": mode,
            "blend_beta": beta,
            "holdout_ndcg@10": holdout,
            "seconds": time.perf_counter() - started,
            "per_user_ndcg": values.tolist(),
        }
        print(
            f"seed {seed} {name} NDCG@10={metrics['ndcg@10']:.6f} beta={beta}",
            flush=True,
        )
    delta = np.asarray(result["two_tower_rp3"]["per_user_ndcg"]) - np.asarray(result["two_tower"]["per_user_ndcg"])
    result["paired_delta_vs_retrained"] = bootstrap(delta)
    published = np.asarray(json.loads(REPORT.read_text(encoding="utf-8"))["published"]["per_user_ndcg"])
    result["paired_delta_vs_published"] = bootstrap(
        np.asarray(result["two_tower_rp3"]["per_user_ndcg"]) - published
    )
    return result


def published_per_user(context, users, positives, ground_truth) -> dict:
    pipeline = bench.make_pipeline(context)
    recommendations = {}
    started = time.perf_counter()
    for done, user_id in enumerate(users, start=1):
        recommendations[user_id] = [int(row["item_id"]) for row in pipeline.recommend(user_id, top_k=10)]
        if done % 400 == 0:
            print(f"published {done}/{len(users)}", flush=True)
    values = per_user(recommendations, users, positives)
    metrics = evaluate_model(recommendations, ground_truth, 10)
    print(f"published NDCG@10={metrics['ndcg@10']:.6f} seconds={time.perf_counter() - started:.1f}", flush=True)
    return {
        "metrics": metrics,
        "matches_recorded_baseline": abs(metrics["ndcg@10"] - PUBLISHED_NDCG) < 1e-12,
        "per_user_ndcg": values.tolist(),
        "seconds": time.perf_counter() - started,
    }


def per_user(recommendations: dict, users: list[int], positives: dict) -> np.ndarray:
    return np.array(
        [ndcg_at_k(recommendations[user_id], {positives[user_id]}, 10) for user_id in users],
        dtype=np.float64,
    )


def bootstrap(delta: np.ndarray, draws: int = 10_000, seed: int = 42) -> dict:
    generator = np.random.default_rng(seed)
    index = generator.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "draws": draws,
        "n_users": int(len(delta)),
    }


def summarize(report: dict, users: list[int]) -> dict:
    seeds = [report["seeds"][str(seed)] for seed in SEEDS]
    rp3 = np.array([row["two_tower_rp3"]["metrics"]["ndcg@10"] for row in seeds])
    tower = np.array([row["two_tower"]["metrics"]["ndcg@10"] for row in seeds])
    published = report["published"]["metrics"]["ndcg@10"]
    return {
        "seeds": list(SEEDS),
        "n_users": len(users),
        "two_tower_retrained": interval(tower),
        "two_tower_rp3": interval(rp3),
        "delta_vs_retrained": interval(rp3 - tower),
        "delta_vs_published_point": interval(rp3 - published),
        "published_ndcg@10": published,
        "seed42_user_bootstrap_vs_retrained": report["seeds"]["42"]["paired_delta_vs_retrained"],
        "seed42_user_bootstrap_vs_published": report["seeds"]["42"]["paired_delta_vs_published"],
    }


def interval(values: np.ndarray) -> dict:
    from scipy import stats

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    se = std / np.sqrt(len(values))
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    return {
        "n": int(len(values)),
        "values": [float(value) for value in values],
        "mean": mean,
        "std": std,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "t_critical": critical,
    }


def save(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(bench.clean(report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
