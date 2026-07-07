# Project Context: RecSys Two-Tower MovieLens-1M

## Requirements Summary
- **R1. Data Pipeline**: Load MovieLens-1M, temporal split (last interaction test, second-last validation, rest train per user), filter users with < 5 ratings. No random split.
- **R2. Two-Stage Recommendation Pipeline**:
  - Stage 1 (Retrieval): In-batch negative sampling, top-200 candidates per query via vector search (FAISS). Index rebuilt on retraining.
  - Stage 2 (Re-ranking): GBDT re-ranker on top-200 candidates to get top-10. Min 8 features. Positives: rating >= 4. Negatives: unrated retrieved items.
- **R3. Cold-Start & API Latency**: API handles unknown `user_id` (content-based textual embeddings recommendation fallback) and unknown `item_id` (graceful HTTP 404, not 500). Response time <= 500ms on CPU.
- **R4. Offline Evaluation**: `evaluate.py` calculating Precision@10, Recall@10, NDCG@10 on test split. Compare (1) popularity baseline, (2) retrieval-only, (3) full pipeline. Reproducible.
- **R5. HTTP Service & UI**: Web UI + API. UI shows watch history and personalized recommendations. Docker Compose launches everything.
- **R6. Testing & Docs**: Unit tests for temporal split, feature count, cold-start fallback, and API endpoint. README with architecture, metrics table, and run instructions.

## Key Directories and Paths
- Working Directory: `z:/pet-project/recsys-two-tower`
- Orchestrator Directory: `z:/pet-project/recsys-two-tower/.agents/orchestrator`
- Original User Request: `z:/pet-project/recsys-two-tower/ORIGINAL_REQUEST.md`

## Acceptance Criteria Thresholds
- Retrieval Recall@200 >= 0.40
- Full pipeline NDCG@10 >= 0.20
- Full pipeline Precision@10 >= 0.15
- Re-ranker NDCG > Retrieval-only NDCG
- Full pipeline NDCG > Popularity baseline NDCG
- GET /recommend/999999 (non-existent user) -> 200 or 404
- GET /health -> 200
- Latency <= 500ms
