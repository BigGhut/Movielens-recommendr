## 2026-07-06T12:04:09Z

You are the Worker (teamwork_preview_worker).
Your working directory is z:/pet-project/recsys-two-tower/.agents/worker_m1/.
Your task is to implement the MovieLens-1M Data Pipeline (Milestone 1) in the workspace.
Please use the following details from the Explorer analysis:
1. Implement `src/data/download.py` to download MovieLens-1M from `http://files.grouplens.org/datasets/movielens/ml-1m.zip` and extract it to `data/raw/ml-1m/`. The download must use the Python standard library (`urllib.request` and `zipfile`) and support a local cache check (skip download if files already exist).
2. Implement `src/data/preprocess.py` to load raw files, filter out users with < 5 interactions (although ml-1m has >= 20 ratings per user, write the filter logic explicitly), perform a temporal train/validation/test split for each user chronologically (deterministically sorting by ['user_id', 'timestamp', 'movie_id'] to break ties), merge user/movie metadata with the splits, and save the split datasets to `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv`.
3. Implement `src/data/loader.py` to provide a utility data loader class for downstream models.
4. Implement `tests/test_data.py` with pytest tests verifying:
   - User filtering integrity.
   - Split shape and boundaries (exactly 1 val, 1 test record per user).
   - No overlap between train, val, and test splits.
   - Strict temporal sequence (no temporal leakage).
5. Verify the code:
   - Run the download script.
   - Run the preprocessing script.
   - Run the tests using pytest.
   Document the commands run and their output in your handoff report.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please write your implementation report to `z:/pet-project/recsys-two-tower/.agents/worker_m1/changes.md` and your handoff report to `z:/pet-project/recsys-two-tower/.agents/worker_m1/handoff.md`. Send a message when complete.
