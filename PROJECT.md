# Project: Two-Tower RecSys MovieLens-1M

## Architecture
An end-to-end two-stage recommendation system:
1. **Data Pipeline (`src/data`)**: Loads MovieLens-1M, filters out users with < 5 interactions, performs a temporal split (train, validation, test) per user, and formats data for model training.
2. **Retrieval Tower (`src/retrieval`)**: User-Tower and Item-Tower PyTorch neural networks trained with in-batch negatives. Rebuilds and indexes item embeddings using FAISS.
3. **Re-ranking GBDT (`src/reranking`)**: Uses LightGBM/CatBoost to score and rank top-200 retrieved items to top-10. Utilizes >= 8 features (user, item, user-item cross features).
4. **HTTP API (`src/api`)**: Serves recommendations at `/recommend/{user_id}` and `/health`. Handles cold start.
5. **Web UI (`src/ui`)**: Renders recommendations and user history.
6. **E2E/Offline Evaluation (`evaluate.py`)**: Computes Precision@10, Recall@10, and NDCG@10 on held-out test split for baseline, retrieval, and full pipeline.

```
+-------------------------------------------------------------+
|                        Data Pipeline                        |
|              (MovieLens-1M -> Temporal Split)               |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       Retrieval Stage                       |
|               (Two-Tower PyTorch Embeddings)                |
|                    FAISS Candidate Index                    |
+------------------------------+------------------------------+
                               | (Top 200 candidates)
                               v
+-------------------------------------------------------------+
|                      Re-ranking Stage                       |
|                   (LightGBM Classifier)                     |
|           (User, Movie, Interaction Features)               |
+------------------------------+------------------------------+
                               | (Top 10 sorted movies)
                               v
+-------------------------------------------------------------+
|                      HTTP API & UI                          |
|             (FastAPI Service, Docker Compose)               |
+-------------------------------------------------------------+
```

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Data Pipeline | Temporal split logic, data loading, user filtering | None | PLANNED |
| 2 | M2: Retrieval Model | PyTorch Two-Tower, in-batch negative loss, FAISS indexing | M1 | PLANNED |
| 3 | M3: Re-ranking Model | LightGBM dataset creation, GBDT model training, >= 8 features | M1, M2 | PLANNED |
| 4 | M4: HTTP API & UI | FastAPI service, cold-start handling, UI client, Docker Compose | M2, M3 | PLANNED |
| 5 | M5: E2E Eval & Tests | evaluate.py implementation, unit/integration test suites | M1, M2, M3, M4 | PLANNED |

## Interface Contracts

### Data Loader ↔ Retrieval & Re-ranking
- **Inputs**: Raw MovieLens-1M files (`ratings.dat`, `users.dat`, `movies.dat`).
- **Outputs**: Train, validation, and test pandas DataFrames / CSVs containing `user_id`, `movie_id`, `rating`, `timestamp`, `title`, `genres`, etc.
- **Contract**: No rating in test-split may have a timestamp earlier than any rating in train/validation split for the same user.

### Retrieval Model ↔ FAISS Indexer
- **Inputs**: Item embedding tensors of size `[num_items, embedding_dim]`.
- **Outputs**: FAISS index file (`item_index.faiss`) and movie ID mapping.
- **Contract**: Index must be rebuilt from scratch upon training, matching the current model version.

### Candidate Generator ↔ Re-ranking Model
- **Inputs**: `user_id` (integer) or user history features.
- **Outputs**: List of top-200 movie candidates (movie IDs) and their retrieval scores.

### Re-ranking Model ↔ API
- **Inputs**: `user_id`, list of 200 candidate movie IDs.
- **Outputs**: List of top-10 movie IDs sorted by score.

## Code Layout
- `src/data/`: Data loading and temporal split scripts.
- `src/retrieval/`: Retrieval training and FAISS candidate indexing.
- `src/reranking/`: GBDT training and feature engineering.
- `src/api/`: FastAPI server and cold-start fallback handlers.
- `src/ui/`: UI dashboard (Streamlit or simple HTML/JS).
- `tests/`: Unit tests.
- `evaluate.py`: Offline evaluation runner.
- `docker-compose.yml`: Multi-container deployment configuration.
