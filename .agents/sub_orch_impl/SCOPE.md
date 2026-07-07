# Scope: Implementation Track

## Architecture
- **Data Pipeline**: Loads MovieLens-1M, filters users with < 5 interactions, performs temporal user-level split, and outputs CSVs/DataFrames to `data/processed/`.
- **Retrieval Model**: Two-tower PyTorch network (user/item towers), trains on train-split using in-batch negative sampling, creates FAISS candidate index of items, saves model checkpoints and FAISS index.
- **Re-ranking Model**: GBDT model (LightGBM/CatBoost) using >= 8 engineered features (user historical stats, movie stats, user-item cross-features). Positives are ratings >= 4, negatives are from retrieval candidates.
- **HTTP API & UI**: FastAPI service hosting `/recommend/{user_id}` and `/health` endpoints. Implements cold-start user/item fallbacks. Streamlit UI interface. Orchestrated via Docker Compose.
- **E2E Integration & Verification**: Runs offline evaluation using `evaluate.py` to compare popularity, retrieval-only, and full-pipeline. Integrates with the E2E test suite when `TEST_READY.md` is published, followed by Tier 5 white-box adversarial coverage hardening.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Data Pipeline | Load MovieLens-1M, filter < 5 ratings, temporal split, save datasets | None | IN_PROGRESS |
| 2 | M2: Retrieval Model | PyTorch Two-Tower, in-batch negative loss, FAISS candidate indexing | M1 | PLANNED |
| 3 | M3: Re-ranking Model | LightGBM/CatBoost training, >= 8 features, dataset generation | M1, M2 | PLANNED |
| 4 | M4: HTTP API & UI | FastAPI service, cold-start handling, Streamlit UI, Docker Compose | M2, M3 | PLANNED |
| 5 | M5: E2E Integration | Verify E2E tests, run Tier 5 white-box adversarial hardening | M1, M2, M3, M4 | PLANNED |

## Interface Contracts

### M1 ↔ M2 (Data Pipeline to Retrieval)
- **Inputs**: Raw MovieLens-1M files in `data/raw/ml-1m/`.
- **Outputs**: `data/processed/train.csv`, `data/processed/validation.csv`, `data/processed/test.csv`.
- **Contract**: Users with < 5 interactions are removed. For each user, the latest interaction goes to test, the second-to-latest goes to validation, and all others go to train.

### M2 ↔ M3 (Retrieval to Re-ranking)
- **Inputs**: Processed datasets, retrieval model checkpoints.
- **Outputs**: `models/retrieval_user.pt`, `models/retrieval_item.pt`, `models/item_index.faiss`, and movie ID mapping.
- **Contract**: Re-ranking uses the top-200 candidates retrieved from the FAISS index for each user to generate training negatives.

### M3 ↔ M4 (Re-ranking to API & UI)
- **Inputs**: Retrieval models/index, GBDT model checkpoint `models/reranker.lgb`.
- **Outputs**: Re-ranking model scoring predictions.
- **Contract**: FastAPI server queries the retrieval model + FAISS index to get 200 candidates, extracts features, runs GBDT, and returns the top-10 sorted movies.

### API ↔ Web UI / Users
- **Endpoint**: `/recommend/{user_id}` or `POST /recommend`.
- **Latencies**: Total request latency <= 500 ms.
- **Cold start**: Non-existent users receive fallback content-based recommendations. Non-existent item requests return HTTP 404.
