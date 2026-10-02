import math

import numpy as np


def precision_at_k(predicted: list, actual: set, k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for item in predicted_k if item in actual)
    return hits / k

def recall_at_k(predicted: list, actual: set, k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for item in predicted_k if item in actual)
    return hits / len(actual)

def ndcg_at_k(predicted: list, actual: set, k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(predicted: list, actual: set, k: int) -> float:
    if not actual:
        return 0.0
    predicted_k = predicted[:k]
    hits = 0
    sum_precs = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual:
            hits += 1
            sum_precs += hits / (i + 1.0)
    return sum_precs / min(len(actual), k)

def evaluate_model(recommendations: dict[int, list], ground_truth: dict[int, set], k: int) -> dict:
    """
    Вычисляет средние метрики для всех пользователей.
    """
    precisions = []
    recalls = []
    ndcgs = []
    maps = []
    
    for user_id, actual in ground_truth.items():
        if not actual:
            continue
        predicted = recommendations.get(user_id, [])
        precisions.append(precision_at_k(predicted, actual, k))
        recalls.append(recall_at_k(predicted, actual, k))
        ndcgs.append(ndcg_at_k(predicted, actual, k))
        maps.append(map_at_k(predicted, actual, k))
        
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0
    }
