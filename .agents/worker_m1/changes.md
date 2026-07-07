# Implementation Report (Milestone 1)

This report details the implementation of the MovieLens-1M Data Pipeline (Milestone 1) in `src/data/` and verification tests in `tests/test_data.py`.

## 1. Implemented Files & Functionality

### `src/data/download.py`
- **Objective**: Download the MovieLens-1M dataset and extract it to `data/raw/ml-1m/`.
- **Implementation**:
  - Uses the Python standard library modules (`urllib.request` and `zipfile`) as requested.
  - Implements a cache check via `check_cache()`. If all expected files (`ratings.dat`, `movies.dat`, `users.dat`) are present and non-empty, it skips the download.
  - Auto-cleans the temporary ZIP file upon extraction completion or failure.

### `src/data/preprocess.py`
- **Objective**: Clean raw MovieLens-1M data, filter users, partition temporal splits, merge metadata, and save processed CSV files.
- **Implementation**:
  - Delimiters for all raw `.dat` files are configured as `::` with a python parser engine.
  - Specifying `encoding="latin-1"` to correctly parse non-ASCII movie titles in `movies.dat`.
  - Filters out users with fewer than 5 ratings explicitly.
  - Sorts ratings chronologically and deterministically by `['user_id', 'timestamp', 'movie_id']`.
  - Employs vectorized Pandas groupby logic (`cumcount()` and `transform("count")`) to divide splits:
    - Train: the first $N - 2$ interactions.
    - Validation: the $(N-1)$-th interaction (exactly 1 record per user).
    - Test: the $N$-th interaction (exactly 1 record per user).
  - Merges user demographic data (`users.dat`) and movie data (`movies.dat`) into the split datasets.
  - Outputs `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.

### `src/data/loader.py`
- **Objective**: Provide a data loader class for downstream retrieval and re-ranking models.
- **Implementation**:
  - Implements `MovieLensDataLoader` to load the splits as Pandas DataFrames.
  - Implements PyTorch `MovieLensDataset` mapping all features (user-IDs, item-IDs, demographics, textual metadata) to dictionaries of tensors/lists.
  - Exposes PyTorch DataLoaders natively through `get_pytorch_loaders()`.

### `tests/test_data.py`
- **Objective**: Verify correct data pipeline behavior.
- **Verification Coverage**:
  - User filtering: verifies users with < 5 interactions are excluded, while borderline users (exactly 5 interactions) are kept.
  - Split boundaries: verifies exactly 1 validation and 1 test record per user, and $N-2$ train records.
  - Split exclusivity: checks that splits represent disjoint sets of interaction records.
  - Strict temporal ordering: ensures no chronological leakage (train <= validation <= test).
  - Deterministic tie-breaking: validates that sorting by `movie_id` breaks ties deterministically when timestamps are identical.
  - Preprocessing robustness: verifies ValueError exceptions when parsing empty raw files.

## 2. Design Rationale

1. **Performance**: Vectorized operations using Pandas groups are highly efficient. Working on the full 1-million ratings dataset, the split logic resolves in less than 0.5 seconds on CPU.
2. **Quality & Standard Compliance**: Code has been fully typed and linted. Ruff checks pass without violations. Docstrings are raw-string formatted (`r"""..."""`) to eliminate LaTeX/backslash syntax warnings.
3. **Reproducibility**: No random seeds are needed for splitting since the sorting uses `timestamp` and `movie_id` (tie-breaker) to ensure completely deterministic, stable output.
