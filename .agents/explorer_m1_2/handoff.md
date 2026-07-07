# Handoff Report: Milestone 1 Data Pipeline Investigation

## 1. Observation

- **Directory State**: We ran the `find_by_name` search tool on `z:/pet-project/recsys-two-tower` (excluding agent directories) and found only two files in the root:
  - `ORIGINAL_REQUEST.md` (Size: 8963 bytes)
  - `PROJECT.md` (Size: 4553 bytes)
  No `src/` or `tests/` directories exist, meaning we are designing the initial layout from scratch.

- **Interface Contract**: In `PROJECT.md`, lines 50-53 state:
  ```text
  50: ### Data Loader ↔ Retrieval & Re-ranking
  51: - **Inputs**: Raw MovieLens-1M files (`ratings.dat`, `users.dat`, `movies.dat`).
  52: - **Outputs**: Train, validation, and test pandas DataFrames / CSVs containing `user_id`, `movie_id`, `rating`, `timestamp`, `title`, `genres`, etc.
  53: - **Contract**: No rating in test-split may have a timestamp earlier than any rating in train/validation split for the same user.
  ```

- **Requirements Context**: In `orchestrator/context.md`, line 4 states:
  ```text
  4: - **R1. Data Pipeline**: Load MovieLens-1M, temporal split (last interaction test, second-last validation, rest train per user), filter users with < 5 ratings. No random split.
  ```

---

## 2. Logic Chain

1. **Clean Canvas Discovery**: Since `src/data/` and `tests/` directories do not exist, we must design a complete layout and files from scratch (referenced in **Directory State** observation).
2. **Filtering Logic**: According to R1 (referenced in **Requirements Context**), users with fewer than 5 interactions must be removed. By executing `value_counts()` on ratings, filtering the index for counts $\ge 5$, and slicing ratings with `.isin()`, we can drop low-interaction users efficiently in a vectorized manner before splitting.
3. **Deterministic Splitting Strategy**: To satisfy the contract that no test rating has an earlier timestamp than train/val for the same user (referenced in **Interface Contract**), interactions must be sorted chronologically per user. Since multiple ratings can share the same timestamp (Unix epoch seconds resolution), sorting by `(timestamp, movie_id)` ensures a deterministic, repeatable order and prevents split leakage.
4. **Vectorized Split Algorithm**: Instead of iterating over users in a loop (which is slow for 1M rows), grouping the sorted ratings by `user_id` and using `cumcount()` allows calculating sequential rank. By computing the reverse rank (`total - 1 - cumcount`), the latest interaction has rank `0` (test), second-latest has rank `1` (val), and others have rank `\ge 2` (train).
5. **Leakage & Split Integrity Tests**: To guarantee that the split works correctly, unit tests must verify:
   - User filtering: drop users with < 5 interactions.
   - User set consistency: train, val, and test splits have identical user sets.
   - Mutual exclusivity: no overlapping interaction pairs between train, val, and test.
   - Temporal sequence: $T_{test} \ge T_{val} \ge T_{train}$ for each user.
   We designed these tests in a synthetic mock-data fixture to allow fast, isolated execution.

---

## 3. Caveats

- **No Network Execution**: Due to network isolation constraints (CODE_ONLY mode), the download script `download.py` could not be run to fetch the actual MovieLens-1M dataset. The logic was verified theoretically.
- **Item Cold-Start**: The user filtering logic does not filter movies. Some movies might only appear in the validation or test splits, not in the training split. The retrieval stage will need content-based fallback or popularity baseline handling for these cold movies.
- **Timestamp Resolution**: MovieLens timestamps are in seconds. In case of identical timestamps, we sort by `movie_id` ascending. If a user bulk-rates multiple items in the same second, the assignment of test vs validation is determined by `movie_id`. This maintains deterministic partitions but represents a minor arbitrary boundary.

---

## 4. Conclusion

The data pipeline design successfully meets all requirements of Milestone 1. The proposed layout is:
- `src/data/download.py` (handles fetching and extraction via standard libraries).
- `src/data/preprocess.py` (handles vectorized loading, user filtering, temporal splitting, and metadata enrichment).
- `tests/test_data_pipeline.py` (contains 6 unit tests validating filtering, sizes, exclusivity, and temporal ordering using synthetic mock datasets).

All templates are documented in `analysis.md` and are ready for implementation by the worker agent.

---

## 5. Verification Method

Once implemented, the pipeline can be verified by running the following command in the project root:
```bash
pytest tests/test_data_pipeline.py
```
This test suite runs instantly using synthetic mock datasets. It verifies that user filtering, temporal split rules, and deterministic sorting are implemented correctly.
Invalidation conditions for this verification:
- Any unit test failure (e.g., temporal leakage detected, wrong split sizes, user set mismatch).
- Preprocessing script failing to run on the real MovieLens-1M dataset due to file encoding issues (must use `ISO-8859-1`).
