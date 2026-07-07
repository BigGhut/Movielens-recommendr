# Milestone 1 Quality and Adversarial Review Report

## Review Summary

**Verdict**: APPROVE

We reviewed the Data Pipeline (Milestone 1) implementation, which includes `src/data/download.py`, `src/data/preprocess.py`, `src/data/loader.py`, and `tests/test_data.py`. All unit tests and E2E baseline tests pass, and the code meets correctness, logical completeness, and safety (no-leakage) requirements. There are a few minor formatting/style findings, which do not block approval.

---

## Findings

### Minor Finding 1: Python Code Formatting Compliance
- **What**: Python source files do not fully conform to the standard Ruff formatting style (e.g., quote styles, blank line placement, wrapping).
- **Where**:
  - `src/data/download.py`
  - `src/data/preprocess.py`
  - `src/data/loader.py`
  - `tests/test_data.py`
- **Why**: Minor style non-conformance. Running `ruff format --check` reports that these 4 files would be reformatted.
- **Suggestion**: Run `ruff format src/data tests/test_data.py` to fix formatting discrepancies.

### Minor Finding 2: Unused Import in E2E Mock Scripts
- **What**: Unused import `os` is present in E2E mock scripts.
- **Where**:
  - `tests/e2e/mock_preprocess.py:1`
  - `tests/e2e/mock_retrieval_train.py:1`
- **Why**: Triggers Ruff lint warning `F401`.
- **Suggestion**: Remove `import os` from these files.

---

## Verified Claims

- **User filtering logic** -> Verified via code review of `filter_users()` and unit test `test_user_filtering` -> **PASS**
  - Users with `< 5` interactions are successfully excluded.
- **Chronological temporal split logic** -> Verified via code review of `temporal_split()` and unit test `test_temporal_split_logic` -> **PASS**
  - Ratings are sorted chronologically by `["user_id", "timestamp", "movie_id"]`.
  - The last rating is placed in `test`, the second-to-last in `val`, and all earlier in `train`.
- **Deterministic tie-breaking** -> Verified via code review and unit test `test_identical_timestamps` -> **PASS**
  - Ties in timestamps are broken deterministically by sorting on `movie_id`.
- **Split exclusivity & no-leakage contract** -> Verified via logic tracing and unit test `test_split_exclusivity` -> **PASS**
  - The train, validation, and test splits are disjoint.
  - Since rows are sorted ascendingly by timestamp, no test rating has a timestamp earlier than any train or validation rating for the same user.
- **Split proportions** -> Verified via unit test `test_split_proportions` -> **PASS**
  - Validation and test sets contain exactly 1 record per user.
- **Empty input handling** -> Verified via unit test `test_empty_ratings_input` -> **PASS**
  - Clean `ValueError` is raised when raw input files are empty.
- **PyTorch Dataset and DataLoader integrations** -> Verified via `test_preprocess_and_save_workflow` -> **PASS**
  - Loader successfully outputs PyTorch `Dataset` and `DataLoader` instances with correct types, shapes, and metadata mapping.

---

## Coverage Gaps

- None identified. The code coverage for the data pipeline is high, and edge cases (borderline ratings, identical timestamps, empty inputs) are fully handled and covered by tests. Risk level is low.

---

## Unverified Items

- **Quantitative evaluation on full MovieLens-1M dataset**: While the pipeline and split logic were verified on mock and test sets, we did not execute the full-scale preprocessing script on the complete MovieLens-1M dataset during this code-level review. However, E2E tests verify the setup.

---

# Adversarial Review (Challenge Report)

## Challenge Summary

**Overall risk assessment**: LOW

---

## Challenges

### Low Challenge 1: Assumption on Minimum Interactions Count
- **Assumption challenged**: The pipeline assumes users always have $\ge 3$ interactions (guaranteed by the default `min_interactions=5` constraint).
- **Attack scenario**: If `min_interactions` is set to `< 3` (e.g. 2 or 1) by a developer, the `temporal_split` logic:
  ```python
  train_mask = cum_count < (group_size - 2)
  val_mask = cum_count == (group_size - 2)
  test_mask = cum_count == (group_size - 1)
  ```
  will result in empty train/val splits for users with 1 or 2 interactions.
- **Blast radius**: Downstream retrieval/ranking models might crash during training due to empty tensors or divisions by zero when handling users with empty histories.
- **Mitigation**: Add an explicit check in `temporal_split` that filters out or warns about users with `group_size < 3`, or raise a descriptive exception.

### Low Challenge 2: MovieLens-1M Text Encoding Limitations
- **Assumption challenged**: Raw MovieLens-1M text files are read assuming `"latin-1"` encoding.
- **Attack scenario**: If the dataset is updated or customized with characters that cannot be decoded under `latin-1` or have malformed byte sequences, the preprocessing script will fail.
- **Blast radius**: Preprocessing script crashes on loading raw files.
- **Mitigation**: Use `errors="replace"` or implement automatic encoding detection for the source files.

---

## Stress Test Results

- **Identical timestamps stress test**: Users with identical timestamps across all ratings are split deterministically using `movie_id` as the tie-breaker -> **PASS** (verified via `test_identical_timestamps`).
- **Empty input stress test**: Empty raw ratings file raises clear validation error -> **PASS** (verified via `test_empty_ratings_input`).

---

## Unchallenged Areas

- FAISS index rebuild efficiency, user/item embedding alignment, re-ranking scoring latency. These belong to Milestones 2-4 and are out of scope for the Milestone 1 (Data Pipeline) review.
