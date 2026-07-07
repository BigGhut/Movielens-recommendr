# Handoff Report: MovieLens-1M Data Pipeline Design

## 1. Observation
- Root project directory contains only `PROJECT.md` and `ORIGINAL_REQUEST.md`. No existing source files, test suites, or download files exist.
- Per `PROJECT.md` (lines 68-74):
  ```
  68: ## Code Layout
  69: - `src/data/`: Data loading and temporal split scripts.
  70: - `src/retrieval/`: Retrieval training and FAISS candidate indexing.
  71: - `src/reranking/`: GBDT training and feature engineering.
  72: - `src/api/`: FastAPI server and cold-start fallback handlers.
  73: - `src/ui/`: UI dashboard (Streamlit or simple HTML/JS).
  74: - `tests/`: Unit tests.
  ```
- Per `PROJECT.md` (lines 48-53):
  ```
  48: ## Interface Contracts
  49: 
  50: ### Data Loader ↔ Retrieval & Re-ranking
  51: - **Inputs**: Raw MovieLens-1M files (`ratings.dat`, `users.dat`, `movies.dat`).
  52: - **Outputs**: Train, validation, and test pandas DataFrames / CSVs containing `user_id`, `movie_id`, `rating`, `timestamp`, `title`, `genres`, etc.
  53: - **Contract**: No rating in test-split may have a timestamp earlier than any rating in train/validation split for the same user.
  ```
- GroupLens documentation indicates that MovieLens-1M is formatted as double-colon `::` separated value files. `movies.dat` contains non-ASCII text which will trigger decoding errors unless parsed with an encoding like `ISO-8859-1` or `latin-1`.
- GroupLens documentation specifies that all users in MovieLens-1M have a minimum of 20 ratings.

## 2. Logic Chain
- Given the empty code layout observation, the implementation must build the structure `src/data/` and `tests/` from scratch.
- The interface contract requires CSV/DataFrame outputs containing demographic, movie, and rating attributes merged together. Thus, the loader should return merged user, movie, and rating features.
- Given the `::` separator and character encoding observations, the data parser must use `engine='python'` and `encoding='ISO-8859-1'` inside `pandas.read_csv`.
- To avoid data leakage and respect the temporal constraint ("no test rating timestamp earlier than train/val for a user"), ratings must be sorted chronologically per user. Since multiple ratings can share the same second timestamp, we must break ties using a secondary key (`movie_id`) to ensure deterministic grouping and ranking.
- Since MovieLens-1M already satisfies the $\ge 5$ interaction threshold per user (users have $\ge 20$ ratings), the user filtering code is not strictly filtering anything out for this dataset, but writing it explicitly ensures the pipeline is robust to other datasets or custom user-sampling.
- Assigning `user_rank` and `user_total` per user in sorted ratings allows exact, vectorized, and deterministic division:
  - Test = last rating (`user_rank == user_total - 1`).
  - Validation = second-to-last rating (`user_rank == user_total - 2`).
  - Train = all prior ratings (`user_rank < user_total - 2`).

## 3. Caveats
- **Same-Second Ratings**: MovieLens timestamps have a resolution of seconds. If a user rates multiple items in the same second, the tie-breaker (`movie_id`) determines which item goes to train vs validation vs test. This ordering is arbitrary, but necessary to keep val and test sizes exactly 1 per user.
- **Scalability**: The vectorized pandas logic holds everything in-memory. While efficient for MovieLens-1M, it may encounter memory issues with larger datasets (e.g. MovieLens-25M or Netflix prize).
- **Cold-Start Handling**: Because users are filtered for $\ge 5$ interactions, every user is guaranteed to have at least 3 train samples, 1 val sample, and 1 test sample. However, new items that were not present in the train split could still appear in val/test splits (item cold-start). This needs to be handled in downstream retrieval/ranking stages.

## 4. Conclusion
We have established a robust download, load, preprocess, split, and unit test design for Milestone 1. The code templates are vectorized, handle text encoding quirks, break tie-breaking timestamps deterministically, and avoid user cold-start in validation/evaluation.

## 5. Verification Method
1. **Verification of Logic**:
   - Confirm that raw data files exist under `data/raw/ml-1m/` after execution.
   - Run the preprocessing script:
     ```powershell
     python src/data/preprocess.py --data-dir data
     ```
   - Verify that `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv` exist.
2. **Pytest Suite Verification**:
   - Run the test suite:
     ```powershell
     pytest tests/test_data.py
     ```
   - Check that all four test targets (filtering, split cardinality, split disjointness, no temporal leakage) pass successfully.
