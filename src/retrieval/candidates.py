import numpy as np


def unseen_candidates(user_emb: np.ndarray, faiss_index, idx2item: dict, history: set, retrieval_k: int) -> list[int]:
    """FAISS order with already seen items removed.

    Searches `retrieval_k * 2` neighbors so filtering history still leaves a full
    candidate list for users with long histories. The rank in this list is the
    retrieval rank used by the ranker.
    """
    if user_emb.ndim == 1:
        user_emb = user_emb.reshape(1, -1)
    _, indices = faiss_index.search(user_emb, retrieval_k * 2)
    candidates = []
    for raw_idx in indices[0]:
        c_idx = int(raw_idx)
        if c_idx < 0 or c_idx not in idx2item:
            continue
        item_id = int(idx2item[c_idx])
        if item_id in history:
            continue
        candidates.append(item_id)
        if len(candidates) >= retrieval_k:
            break
    return candidates
