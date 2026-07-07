# E2E Test Infrastructure & Design

This document details the end-to-end (E2E) testing framework, 4-tier methodology, and test cases for the Two-Tower Recommendation System on MovieLens-1M.

## 1. Test Runner Architecture

The test suite is built on **pytest** and designed as an **opaque-box** testing framework. It interacts with the system using external boundaries: HTTP API endpoints, CLI entry points, and output artifacts (CSVs, model weights, evaluation JSONs).

### Directory Layout
All E2E tests are located in `tests/e2e/`.
```
tests/
├── __init__.py
├── conftest.py                # Shared fixtures (API client, mock data, environment variables)
└── e2e/
    ├── __init__.py
    ├── test_f1_data.py        # F1: Data Preprocessing & Split tests (T1 & T2)
    ├── test_f2_retrieval.py   # F2: Retrieval & FAISS Indexing tests (T1 & T2)
    ├── test_f3_reranking.py   # F3: Re-ranking & Feature Engineering tests (T1 & T2)
    ├── test_f4_cold_start.py   # F4: Cold Start handling tests (T1 & T2)
    ├── test_f5_latency.py     # F5: API Performance Latency tests (T1 & T2)
    ├── test_f6_eval.py        # F6: Offline Evaluation tests (T1 & T2)
    ├── test_f7_api_ui.py      # F7: HTTP API & UI deployment tests (T1 & T2)
    ├── test_combinations.py   # Tier 3: Cross-Feature Integration tests
    └── test_scenarios.py      # Tier 4: Real-world Scenario workflows
```

### Opaque-Box Execution Design
- **Data & Training Pipelines**: Verified by triggering the preprocessing and training scripts via `subprocess` calls and verifying that the outputs on disk (`data/processed/*.csv`, `models/*.pt`, `models/item_index.faiss`) match schema contracts.
- **FastAPI Endpoint**: Tested via an HTTP client (`httpx` or `requests`) targeting a running instance of the API (either local or inside Docker Compose).
- **Offline Evaluation**: Verified by executing `evaluate.py` as a subprocess and loading/parsing the generated JSON metrics file.

### Environment & Test Execution Modes
To guarantee fast local iterations while preserving full-scale validation:
1. **Mock/Fast Mode (`--use-mock-data`)**: Generates a tiny synthetic MovieLens-1M dataset, trains miniature models (embedding size = 4, GBDT max_depth = 2), and indexes them. Tests should complete in under 30 seconds.
2. **Production Mode (Default)**: Runs on the full MovieLens-1M dataset and validates hard acceptance criteria (Recall, NDCG, Precision, Latency).

---

## 2. Core Features Under Test

The recommendation system consists of $N=7$ core features:
* **F1: Data Preprocessing & Temporal Split**
* **F2: Retrieval Model & Candidate Generation & FAISS Indexing**
* **F3: Re-ranking GBDT Model & Feature Engineering (>= 8 features)**
* **F4: Cold Start fallback handling (user and movie)**
* **F5: API Performance Latency (<= 500ms)**
* **F6: Offline Evaluation script (evaluate.py logic & metrics)**
* **F7: HTTP API & UI deployment (docker compose & distinct recommendations)**

---

## 3. The 4-Tier Test Suite Design

### Tier 1: Feature Coverage (35 Tests, 5 per feature)

| Test ID | Feature | Function Name | Description | Key Assertions |
|---|---|---|---|---|
| **T1_F1_1** | F1 | `test_user_filtering` | Verify filtering of low-interaction users. | Users with < 5 interactions are completely excluded. |
| **T1_F1_2** | F1 | `test_temporal_split_logic` | Verify train/val/test temporal sequence. | For each user: $T_{train} \le T_{val} \le T_{test}$. |
| **T1_F1_3** | F1 | `test_split_exclusivity` | Verify no interaction overlap between splits. | Disjoint set of user-movie-timestamp records. |
| **T1_F1_4** | F1 | `test_data_leakage_absence` | Ensure no test interactions are in train set. | No timestamp in test split is earlier than train split. |
| **T1_F1_5** | F1 | `test_split_proportions` | Verify exact counts of validation and test. | Count of val/test records equals number of unique users. |
| **T1_F2_1** | F2 | `test_retrieval_candidate_count` | Verify candidate generation returns 200 items. | Candidate list length == 200. |
| **T1_F2_2** | F2 | `test_faiss_index_rebuild` | Verify indexing rebuilds from scratch. | File write time of `item_index.faiss` updates on train. |
| **T1_F2_3** | F2 | `test_in_batch_negatives` | Verify neural training completes with correct loss. | Loss decreases, output embedding dim matches configuration. |
| **T1_F2_4** | F2 | `test_faiss_retrieval_correctness` | Verify basic relevance of FAISS index search. | Querying user embedding returns matching items. |
| **T1_F2_5** | F2 | `test_embedding_alignment` | Verify user/item embedding dimension alignment. | Embedding sizes match. Dot product evaluates correctly. |
| **T1_F3_1** | F3 | `test_reranking_output_count` | Verify GBDT ranks and returns top-10 items. | Output recommendation list length == 10. |
| **T1_F3_2** | F3 | `test_feature_count_minimum` | Check that at least 8 features are engineered. | Feature matrix shape column count >= 8. |
| **T1_F3_3** | F3 | `test_gbdt_training_loss` | Verify GBDT training completes successfully. | Validation log-loss/binary-logloss decreases. |
| **T1_F3_4** | F3 | `test_gbdt_scoring_monotonicity` | Verify higher prediction score maps to top rank. | Sorting is monotonically decreasing by score. |
| **T1_F3_5** | F3 | `test_feature_importance` | Ensure all 8+ features show importance > 0. | Feature importance dict has >= 8 keys with non-zero weight. |
| **T1_F4_1** | F4 | `test_cold_user_status` | Verify API request for unknown user returns HTTP 200. | API response code == 200. |
| **T1_F4_2** | F4 | `test_cold_user_fallback` | Verify content-based fallback for cold user. | Recommendations returned based on description text embedding. |
| **T1_F4_3** | F4 | `test_cold_movie_status` | Verify request for unknown movie returns HTTP 404. | API response code == 404 (not 500). |
| **T1_F4_4** | F4 | `test_cold_user_empty_history` | Verify fallback for user with zero ratings. | Returns valid top-10 movie IDs. |
| **T1_F4_5** | F4 | `test_cold_start_unindexed_movie` | Verify API handles newly added unindexed movie. | API ignores or treats gracefully, does not crash. |
| **T1_F5_1** | F5 | `test_latency_standard_user` | Verify API latency for typical user <= 500ms. | Response time <= 500ms on CPU. |
| **T1_F5_2** | F5 | `test_latency_cold_user` | Verify API latency for cold user fallback <= 500ms. | Response time <= 500ms on CPU. |
| **T1_F5_3** | F5 | `test_latency_concurrent_light` | Verify latency under 5 concurrent users. | Mean response time <= 500ms. |
| **T1_F5_4** | F5 | `test_latency_health_check` | Verify health check latency <= 50ms. | Response time <= 50ms. |
| **T1_F5_5** | F5 | `test_latency_sequential` | Verify response time over 20 sequential queries. | Every request resolves in <= 500ms. |
| **T1_F6_1** | F6 | `test_eval_metrics_present` | Verify evaluate.py outputs JSON with required metrics. | Keys `Precision@10`, `Recall@10`, `NDCG@10` in JSON. |
| **T1_F6_2** | F6 | `test_eval_model_comparison` | Verify evaluation compares all 3 baseline models. | Popularity, Retrieval-Only, Full-Pipeline present in JSON. |
| **T1_F6_3** | F6 | `test_eval_reproducibility` | Verify identical outputs on run with same seed. | Metrics are byte-identical across runs. |
| **T1_F6_4** | F6 | `test_eval_metric_bounds` | Verify metrics values are within bounds. | Metrics $\in [0.0, 1.0]$. |
| **T1_F6_5** | F6 | `test_eval_test_exclusivity` | Verify evaluation evaluates only on test split. | Evaluation uses `test.csv` interactions exclusively. |
| **T1_F7_1** | F7 | `test_api_health_ok` | Verify API health endpoint is healthy. | `GET /health` body contains `{"status": "healthy"}`. |
| **T1_F7_2** | F7 | `test_docker_compose_up` | Verify Docker containers launch and bind port. | Containers run, port 8000 and 8501 are reachable. |
| **T1_F7_3** | F7 | `test_distinct_recommendations` | Verify recommendations vary across different users. | Recs list for User A != Recs list for User B. |
| **T1_F7_4** | F7 | `test_ui_elements_present` | Verify Web UI layout renders expected components. | UI shows history table and top-10 recommended table. |
| **T1_F7_5** | F7 | `test_api_cors` | Verify CORS headers are present on API responses. | `Access-Control-Allow-Origin` header exists. |

---

### Tier 2: Boundary & Corner Cases (35 Tests, 5 per feature)

| Test ID | Feature | Function Name | Description | Key Assertions |
|---|---|---|---|---|
| **T2_F1_1** | F1 | `test_user_exactly_5_ratings` | Verify split for borderline user with 5 ratings. | 3 in train, 1 in validation, 1 in test. |
| **T2_F1_2** | F1 | `test_user_exactly_4_ratings` | Verify filtering for borderline user with 4 ratings. | User is completely removed from all splits. |
| **T2_F1_3** | F1 | `test_identical_timestamps` | Verify split logic when timestamps are identical. | Deduplicates or splits deterministically (no crash). |
| **T2_F1_4** | F1 | `test_empty_ratings_input` | Verify preprocessing behavior with empty raw file. | Raises clean ValueError, does not crash silently. |
| **T2_F1_5** | F1 | `test_extreme_temporal_range` | Verify correctness when temporal split range is small. | Correctly partitions even if timestamps span single day. |
| **T2_F2_1** | F2 | `test_faiss_empty_index` | Verify querying an uninitialized FAISS index. | Raises descriptive RuntimeError or CustomException. |
| **T2_F2_2** | F2 | `test_fewer_than_200_items` | Verify candidate generation with small catalog. | Returns all available catalog items, no index errors. |
| **T2_F2_3** | F2 | `test_faiss_nan_embeddings` | Verify indexing items with NaN in embedding. | Indexer throws validation error or filters out NaN. |
| **T2_F2_4** | F2 | `test_out_of_bounds_ids` | Verify FAISS returns valid ID range. | All movie IDs returned exist in metadata database. |
| **T2_F2_5** | F2 | `test_large_catalog_scale` | Verify index building on mock 100k items. | Builds successfully, search takes <= 5ms. |
| **T2_F3_1** | F3 | `test_gbdt_missing_features` | Verify GBDT scoring with missing/NaN features. | Model outputs valid scores without throwing NaNs. |
| **T2_F3_2** | F3 | `test_rerank_fewer_candidates` | Verify re-ranker processes < 200 candidates. | Reranks e.g. 50 items down to top-10 correctly. |
| **T2_F3_3** | F3 | `test_rerank_duplicate_candidates` | Verify handling of duplicate candidate IDs. | Deduplicates candidate list before scoring. |
| **T2_F3_4** | F3 | `test_gbdt_constant_features` | Verify GBDT with invariant feature values. | Scores items successfully (no zero-division/exceptions). |
| **T2_F3_5** | F3 | `test_all_negative_users` | Verify training behavior when user has no ratings >= 4. | GBDT training handles user target correctly. |
| **T2_F4_1** | F4 | `test_cold_user_extreme_id` | Verify negative or huge user ID cold start. | ID `-1` or `999999999` falls back to content search. |
| **T2_F4_2** | F4 | `test_cold_movie_non_numeric` | Verify string item ID request. | `GET /movie/invalid` returns HTTP 400 or 404 (not 500). |
| **T2_F4_3** | F4 | `test_cold_user_missing_metadata` | Verify cold start when text descriptions are empty. | Degrades to popularity baseline gracefully. |
| **T2_F4_4** | F4 | `test_cold_api_malformed_body` | Verify API validation of request schema. | Malformed JSON payload returns HTTP 422. |
| **T2_F4_5** | F4 | `test_cold_start_disjoint_users` | Verify testing on completely unseen user population. | API services all unseen test users via fallback. |
| **T2_F5_1** | F5 | `test_latency_high_history_user` | Verify latency for user with 10k interactions. | End-to-end response time <= 500ms. |
| **T2_F5_2** | F5 | `test_latency_under_read_lock` | Verify latency while reading from locked DB files. | File locks do not cause request timeouts. |
| **T2_F5_3** | F5 | `test_latency_first_request_cold` | Verify first-request cold latency handling. | Warm-up routine ensures first request doesn't timeout. |
| **T2_F5_4** | F5 | `test_latency_large_top_k` | Verify latency if requesting top-100 recommendations. | Latency remains <= 500ms. |
| **T2_F5_5** | F5 | `test_latency_during_faiss_reload` | Verify latency while FAISS index updates in background. | Zero request drops, latency spike <= 100ms. |
| **T2_F6_1** | F6 | `test_eval_empty_test_split` | Verify evaluation with empty test dataset. | Returns metrics as 0.0 or raises clean ValueError. |
| **T2_F6_2** | F6 | `test_eval_perfect_predictions` | Verify metrics calculation with perfect model. | Precision, Recall, and NDCG equal exactly 1.0. |
| **T2_F6_3** | F6 | `test_eval_worst_predictions` | Verify metrics calculation with worst-performing model. | All evaluation metrics evaluate to 0.0. |
| **T2_F6_4** | F6 | `test_eval_unseen_movie_in_test` | Verify evaluation on movie not present in training. | Handled as a miss, does not throw index errors. |
| **T2_F6_5** | F6 | `test_eval_write_protected_out` | Verify evaluate.py when output path is read-only. | Logs error cleanly and prints JSON metrics to stdout. |
| **T2_F7_1** | F7 | `test_docker_port_in_use` | Verify Docker setup behavior during port conflict. | Clean container abort with error output (no orphan states). |
| **T2_F7_2** | F7 | `test_db_file_deletion_mid_run` | Verify API state when weight files are deleted. | Graceful API degradation (HTTP 503 or error response). |
| **T2_F7_3** | F7 | `test_ui_empty_input` | Verify Web UI response to blank User ID search. | Shows validation warning, does not send request. |
| **T2_F7_4** | F7 | `test_ui_long_titles` | Verify UI layout with extremely long titles. | Word-wrap handles text; no layout distortion. |
| **T2_F7_5** | F7 | `test_api_unsupported_methods` | Verify calling endpoints with invalid HTTP verbs. | Returns HTTP 405 Method Not Allowed. |

---

### Tier 3: Cross-Feature Combinations (7 Tests)

These tests verify critical interactions between core features.

1. **`test_temporal_split_and_cold_start_integration` (F1 $\leftrightarrow$ F4)**
   - *Logic*: Check that users filtered out during data preprocessing (due to < 5 ratings) are treated as cold-start users by the HTTP API, falling back to content recommendations rather than returning an error.
   - *Key Assertions*: Querying an excluded user ID returns HTTP 200, is marked as fallback, and yields valid recommendations.
2. **`test_faiss_rebuild_and_api_consistency` (F2 $\leftrightarrow$ F7)**
   - *Logic*: Verify that immediately after a model training run completes and overwrites `item_index.faiss`, the running HTTP API reloads the new index space dynamically without dropping active connections.
   - *Key Assertions*: Active requests during reload return HTTP 200. Subsequent recommendations match the updated vector space.
3. **`test_gbdt_and_cold_start_fallback` (F3 $\leftrightarrow$ F4 $\leftrightarrow$ F5)**
   - *Logic*: Check that cold-start users bypass GBDT feature engineering and candidate generation to directly trigger content-based fallback, keeping execution time well under the 500ms CPU SLA.
   - *Key Assertions*: Request resolves successfully, response is marked as content-based fallback, and latency is <= 100ms.
4. **`test_eval_metrics_and_gbdt_features_alignment` (F3 $\leftrightarrow$ F6)**
   - *Logic*: Retrain the GBDT model with only 4 features instead of 8. Verify that `evaluate.py` correctly reports a change (typically a drop) in NDCG@10, proving the evaluation pipeline is sensitive to model quality degradation.
   - *Key Assertions*: Evaluation completes; NDCG@10 for the full stage drops relative to the 8+ feature run.
5. **`test_retrieval_candidate_bound_and_reranking_ndcg` (F2 $\leftrightarrow$ F3 $\leftrightarrow$ F6)**
   - *Logic*: Restrict the candidate generator to feed only 10 candidates to the GBDT (instead of 200). Run the evaluation script.
   - *Key Assertions*: NDCG@10 for the full stage drops significantly, proving that GBDT performance depends on a high-recall retrieval stage.
6. **`test_latency_scaling_with_candidates` (F2 $\leftrightarrow$ F3 $\leftrightarrow$ F5)**
   - *Logic*: Benchmark API latency as candidate counts scale from 10 to 500.
   - *Key Assertions*: E2E Latency remains <= 500ms at 200 candidates; latency scales near-linearly with GBDT scoring load.
7. **`test_temporal_split_and_offline_eval_alignment` (F1 $\leftrightarrow$ F6)**
   - *Logic*: Verify that `evaluate.py` strictly reads the datasets output by the preprocessing split.
   - *Key Assertions*: Test interactions evaluated in `evaluate.py` are byte-identical to `data/processed/test.csv`.

---

### Tier 4: Real-World Application Scenarios (5 Workloads)

These tests simulate realistic workflows representing operational patterns.

#### Scenario 1: Standard User Recommendations Journey
- **Description**: Simulates a registered user logging in and viewing their homepage.
- **Workflow**:
  1. Retrieve the user's historical ratings from `data/processed/train.csv`.
  2. Send a request to `GET /recommend/{user_id}`.
  3. UI retrieves the recommended list.
- **Key Assertions**:
  - Response status code is 200.
  - Latency is <= 500ms.
  - Returned recommendation list has length 10.
  - Recommendations correspond to the user's genre preferences (visual validation).

#### Scenario 2: Dynamic System Training and Redeployment
- **Description**: Simulates a production retrain cycle triggered by new ratings.
- **Workflow**:
  1. Append new interactions to raw data.
  2. Execute preprocess script `python src/data/preprocess.py`.
  3. Train retrieval model `python src/retrieval/train.py`.
  4. Train GBDT model `python src/reranking/train.py` (which rebuilds FAISS index).
  5. Fire concurrent queries to the running API during training and index rebuild.
- **Key Assertions**:
  - Training scripts exit with status 0.
  - FAISS index is updated on disk.
  - API remains online (0 dropped queries, max latency <= 600ms during reloading).

#### Scenario 3: Multi-User Concurrent Traffic Peak
- **Description**: Simulates a traffic spike of concurrent requests.
- **Workflow**:
  1. Spawn 20 concurrent threads using `asyncio` or a load test script.
  2. Mixture of users: 70% registered users, 20% cold-start users, 10% non-existent/invalid IDs.
  3. Send traffic burst to `/recommend/{user_id}`.
- **Key Assertions**:
  - 100% of requests complete successfully (no HTTP 500).
  - Average latency across all requests is <= 500ms.
  - Recommendation lists are distinct for different registered users.

#### Scenario 4: Graceful Degradation under Database / Storage Failure
- **Description**: Simulates system resilience during model or storage corruption.
- **Workflow**:
  1. Temporarily rename or remove `models/item_index.faiss` or GBDT model checkpoint `models/reranker.lgb`.
  2. Send request to `GET /recommend/{user_id}`.
- **Key Assertions**:
  - API does not throw raw Python tracebacks (no HTTP 500).
  - API returns HTTP 503 Service Unavailable or degrades gracefully to a hardcoded popularity list (HTTP 200 with fallback label).

#### Scenario 5: Complete Cold-Start Deployment Verification
- **Description**: Simulates a developer spinning up the stack from absolute scratch.
- **Workflow**:
  1. Run `docker compose down -v` to purge all containers, networks, and volumes.
  2. Execute `docker compose up -d`.
  3. Poll the API health endpoint `/health`.
  4. Once healthy, run a smoke recommendation query.
- **Key Assertions**:
  - Docker Compose exits with status 0.
  - `/health` responds HTTP 200 within 2 seconds of the API service starting.
  - Recommended list is returned successfully.

---

## 4. Test Runner & CLI Usage

All tests can be executed using the `pytest` runner.

### Local Mock Run (Fast Validation)
Executes all E2E tests against synthetic mock data to ensure code compilation, routing, and script integrations work correctly.
```bash
pytest tests/e2e/ --use-mock-data -v
```

### Full Production Verification Run
Executes the test suite against the full MovieLens-1M dataset. This checks the actual quantitative thresholds.
```bash
pytest tests/e2e/ -v
```

### Configuration Variables
The E2E suite reads configuration from environment variables or `pytest.ini`:
- `API_URL`: Target URL for the running API (e.g. `http://localhost:8000`).
- `DATA_DIR`: Path to MovieLens-1M processed split directories.
- `MODEL_DIR`: Path containing trained model weights and FAISS indexes.
