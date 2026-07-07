# Review Report - Milestone 1: Data Pipeline

This report presents the Quality Review and Adversarial Review (Challenges) for the Data Pipeline implementation of the Two-Tower Recommendation System.

---

## Part 1: Quality Review

### Review Summary

**Verdict**: **APPROVE** (with minor code style findings)

The Milestone 1 implementation is correct, conforms to the interface contracts, and passes all unit and integration tests. The split logic successfully isolates the validation and test sets chronologically per user, avoiding data leakage.

---

### Findings

#### [Minor] Finding 1: Unused Imports in Test/Mock Files
- **What**: Ruff lint check failed due to unused import `os`.
- **Where**:
  - `tests/e2e/mock_preprocess.py:1:8`
  - `tests/e2e/mock_reranking_train.py:1:8`
  - `tests/e2e/mock_retrieval_train.py:1:8`
- **Why**: Unused code imports clutter the code and violate strict linter policies.
- **Suggestion**: Remove `import os` from these files.

#### [Minor] Finding 2: Code Formatting Style Discrepancies
- **What**: Ruff format check failed. 11 files would be reformatted.
- **Where**:
  - `src/data/download.py`
  - `src/data/loader.py`
  - `src/data/preprocess.py`
  - `tests/test_data.py`
  - And all files under `tests/e2e/`.
- **Why**: Inconsistent code formatting makes code reviews harder and degrades codebase readability.
- **Suggestion**: Run `ruff format src/ tests/` to automatically format all files to match the style standards.

---

### Verified Claims

- **Claim 1: User filtering removes users with < 5 interactions**
  - *Verified via*: Inspecting `src/data/preprocess.py` line 80-90, running `pytest tests/test_data.py::test_user_filtering` and `pytest tests/test_data.py::test_user_filtering_borderline`.
  - *Result*: **PASS**. Users with fewer than 5 ratings are correctly filtered out.
  
- **Claim 2: Temporal sequence split per user ($T_{train} \le T_{val} \le T_{test}$)**
  - *Verified via*: Inspecting `src/data/preprocess.py` line 110-134, running `pytest tests/test_data.py::test_temporal_split_logic`.
  - *Result*: **PASS**. The split uses pandas `cumcount` on data sorted by `["user_id", "timestamp", "movie_id"]`.
  
- **Claim 3: No overlap between splits (exclusivity)**
  - *Verified via*: Running `pytest tests/test_data.py::test_split_exclusivity`.
  - *Result*: **PASS**. Splits are disjoint.
  
- **Claim 4: Absence of data leakage (no test timestamp is earlier than train/val)**
  - *Verified via*: Code review of the sorting and split logic in `src/data/preprocess.py`. Since sorting is chronological, the last element (test) always has a timestamp greater than or equal to preceding elements (train/val).
  - *Result*: **PASS**.

- **Claim 5: Split proportions (exactly 1 val and 1 test record per user)**
  - *Verified via*: Running `pytest tests/test_data.py::test_split_proportions`.
  - *Result*: **PASS**.

---

### Coverage Gaps

- **Movie IDs containing gaps / missing from train split**
  - *Risk level*: **Medium**.
  - *Description*: Since MovieLens-1M movie IDs go up to 3952 but there are only 3883 movies, and some movies might only be rated once (and thus end up in the test set), the model might encounter unseen movie IDs during evaluation or serving.
  - *Recommendation*: Ensure the Two-Tower model maps IDs to contiguous spaces or allocates a `<UNK>` token for movies not present in the training set.

---

### Unverified Items

- **Download functionality of `src/data/download.py`**
  - *Reason not verified*: Blocked by `CODE_ONLY` network mode constraint which prohibits external network requests during execution. However, the logic was inspected statically and appears correct (checks cache, downloads, extracts, verifies, cleans up).

---

## Part 2: Adversarial Review (Challenges)

### Challenge Summary

**Overall risk assessment**: **LOW**

The data pipeline has been designed with high determinism (e.g., sorting by movie ID to break timestamp ties). However, there are some assumptions about the raw data quality that could cause issues under different inputs.

---

### Challenges

#### [Medium] Challenge 1: Duplicate Ratings Leakage
- **Assumption challenged**: Raw MovieLens data does not contain duplicate ratings for the same `(user_id, movie_id, timestamp)`.
- **Attack scenario**: If raw data contains duplicate rows (e.g., from network retries or database ingestion bugs), the sorting and `cumcount()` logic will partition these duplicate records into different splits (e.g., one in train, one in test). This causes direct data leakage because the model will be evaluated on the exact same interaction it trained on.
- **Blast radius**: High. Evaluated metrics (Precision, NDCG) would be artificially inflated.
- **Mitigation**: Add a deduplication step in `preprocess.py` before splitting:
  ```python
  ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"])
  ```

#### [Low] Challenge 2: Out of Memory (OOM) on Large Scale
- **Assumption challenged**: The entire raw dataset fits comfortably in memory.
- **Attack scenario**: While MovieLens-1M is small (6MB), applying the same script to MovieLens-20M or production clickstreams would exceed system memory due to Pandas' memory footprint during grouping and merging.
- **Blast radius**: Medium (causes process crash).
- **Mitigation**: Use a more scalable processing library like PySpark, Dask, or Polars for larger datasets.

---

### Stress Test Results

- **Tie-breaking identical timestamps**
  - *Expected behavior*: Deterministic split of ratings when timestamps are identical.
  - *Actual behavior*: Verified via `test_identical_timestamps`. The pipeline sorts by `movie_id` to break ties, assigning the movie with the highest ID to the test set, the second highest to validation, and the rest to train.
  - *Result*: **PASS**.

- **Borderline users (exactly 5 ratings and 4 ratings)**
  - *Expected behavior*: User with 4 ratings is filtered out; user with 5 ratings has exactly 3 train, 1 val, 1 test.
  - *Actual behavior*: Verified via `test_user_filtering_borderline`.
  - *Result*: **PASS**.
