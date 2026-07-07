# Challenger Report: Data Pipeline (Milestone 1) Verification

## Challenge Summary

**Overall risk assessment**: **HIGH**

Empirical verification of the Milestone 1 Data Pipeline has revealed a critical silent data corruption vulnerability, along with minor inconsistencies between the production and mock preprocessing pipelines. While the deterministic chronological split works as expected under standard conditions and extreme timestamps, the pipeline lacks robust referential integrity checking, which will lead to downstream crashes during model training.

---

## Challenges

### [High] Challenge 1: Silent PyTorch Tensor Corruption on Unseen Users (Referential Integrity)

- **Assumption challenged**: Raw datasets have complete referential integrity, meaning every user rating in `ratings.dat` has a corresponding record in `users.dat`.
- **Attack scenario**: If a user exists in `ratings.dat` but is missing from `users.dat`, the merge operation in `preprocess.py` creates `NaN` values for `age` and `occupation`. When `MovieLensDataLoader` converts these values to PyTorch long tensors in `MovieLensDataset`, PyTorch silently casts float NaNs on CPU/Windows to `torch.iinfo(torch.long).min` (`-9223372036854775808`) without throwing any warning or error.
- **Blast radius**: The data pipeline exits successfully and saves corrupt files. When the Two-Tower model tries to train, embedding layer lookups for these extremely negative indices will throw out-of-bounds runtime errors, crashing training.
- **Mitigation**:
  - Perform an inner join on metadata or assert that no key columns contain NaNs after merging.
  - Fill missing categorical attributes with a default index (e.g. `0`) or raise a descriptive error in the preprocessing phase.

### [Medium] Challenge 2: Missing Movie Metadata causing NaN Strings

- **Assumption challenged**: Every movie rated in `ratings.dat` exists in `movies.dat`.
- **Attack scenario**: If a movie is rated but missing from `movies.dat`, the left join creates `NaN` fields for `title` and `genres`.
- **Blast radius**: String-based features (e.g., text representations for cold start) will contain float `nan` instead of strings, causing `TypeError` exceptions during sentence embedding generation or tokenization.
- **Mitigation**: Fill missing strings with a default value like `"unknown"` or drop ratings corresponding to unindexed movies during preprocessing.

### [Medium] Challenge 3: Inconsistent Empty File Handling in Mock Preprocessing

- **Assumption challenged**: Production and mock preprocessing scripts share identical validation rules.
- **Attack scenario**: `mock_preprocess.py` does not check file sizes and successfully runs on empty inputs (creating empty CSV outputs and returning exit code `0`). In contrast, `preprocess.py` correctly checks file sizes and raises `ValueError`.
- **Blast radius**: This discrepancy breaks the E2E test `test_empty_ratings_input` when run in mock mode.
- **Mitigation**: Update `mock_preprocess.py` to raise `ValueError` on empty inputs, aligning its behavior with `preprocess.py`.

### [Low] Challenge 4: Empty Train/Val Splits for Low-Interaction Users

- **Assumption challenged**: All users have enough ratings to be partitioned across all splits.
- **Attack scenario**: If `min_interactions` is configured below 3, users with 1 or 2 interactions will have empty train and/or validation splits.
- **Blast radius**: The pipeline runs and the loader compiles, but model training might behave unpredictably or fail due to zero interactions for some users in the training set.
- **Mitigation**: Enforce a constraint where `min_interactions >= 3` or raise an error/warning when users have empty training segments.

---

## Stress Test Results

We implemented and ran five stress-test scenarios in `tests/test_data_challenger.py`. All tests passed, confirming the behavior of the pipeline under boundary conditions:

| Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| **Missing Raw Files** | Raise `FileNotFoundError` | Raised `FileNotFoundError` | **PASS** |
| **Unseen Users** | Detect silent corruption (tensor negative values) | `age` and `occupation` contain `-9223372036854775808` | **PASS (Vulnerability Confirmed)** |
| **Unseen Movies** | Merge NaN titles | `title` field evaluates to float `nan` | **PASS** |
| **Single-Rating Users (min=1)** | Empty train/val splits; test has 1 | Train: 0, Val: 0, Test: 1; compiles successfully | **PASS** |
| **Extreme/Negative Timestamps** | Correct sorting and conversion | Clean ascending partition; stored as `torch.long` | **PASS** |

---

## Unchallenged Areas

- **Full-scale dataset scaling limit**: We did not benchmark preprocess on memory limit since the raw MovieLens-1M dataset is small enough (~24MB raw) to easily fit in RAM.
