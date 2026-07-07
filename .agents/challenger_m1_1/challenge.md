# Adversarial Review Challenge Report - Data Pipeline (Milestone 1)

## Challenge Summary

**Overall risk assessment**: MEDIUM

Milestone 1 (Data Pipeline) implements correct temporal splitting logic per user (with validation and test splits containing exactly 1 chronological record per user and training split containing preceding records) and has no temporal data leakage. However, several critical vulnerabilities and bugs were found during empirical stress testing:
1. **Silent User Metadata Corruption**: Missing user demographics (`users.dat`) are left-joined as NaNs and silently converted to PyTorch long tensors as garbage values (`-9223372036854775808`) during model loading, corrupting training representations without throwing errors.
2. **Movie Metadata Type Mismatch**: Missing movies (`movies.dat`) result in float `NaN` titles and genres in the dataset, which will crash downstream NLP tokenizers/text encoders expecting strings.
3. **Empty Split Crashing**: When any split (e.g. train) is empty due to sparse history, the PyTorch loader crashes with `TypeError` when trying to convert `object` dtypes of the empty pandas DataFrame to long tensors.
4. **Mixed Dtype in zip_code**: The production split contains mixed string and integer zip codes, triggering a `DtypeWarning` in pandas and causing feature engineering instability.
5. **E2E Mock Divergence**: The mock preprocessor (`mock_preprocess.py`) does not check for empty input files and exits with status 0 (producing empty splits), violating test expectations and causing `test_empty_ratings_input` to fail in mock test runs.

---

## Challenges

### [High] Challenge 1: Silent User Metadata Corruption
- **Assumption challenged**: That the raw datasets (`ratings.dat` and `users.dat`) are fully aligned and that a left-join merge during preprocessing is sufficient.
- **Attack scenario**: A user has interactions in `ratings.dat` but is missing from `users.dat`. Preprocessing runs a left-join merge, filling user demographic features (`age`, `occupation`) with `NaN`.
- **Blast radius**: When loading the datasets into PyTorch using `MovieLensDataLoader`, PyTorch silently casts `NaN` floats to the long dtype integer representation (`-9223372036854775808`). The model trains on garbage features without throwing any errors, leading to degraded recommendation quality or model divergency.
- **Mitigation**: Implement inner joins during preprocessing or fill missing/NaN user demographics with a dedicated cold-start placeholder (e.g., `-1`) before converting to long tensor.

### [Medium] Challenge 2: Movie Metadata Type Inconsistency
- **Assumption challenged**: That all rated movies are documented in `movies.dat`.
- **Attack scenario**: A movie ID is present in `ratings.dat` but missing in `movies.dat`. Preprocessing performs a left join, resulting in `NaN` (float) for the `title` and `genres` columns.
- **Blast radius**: The `MovieLensDataset` loads `title` and `genres` as float `NaN` instead of strings. Downstream NLP text encoders, tokenizers, or logging components expecting string metadata will crash.
- **Mitigation**: Add a validation step in preprocessing that checks for orphaned movie IDs, or fill missing titles and genres with placeholder strings like `"Unknown"` or `"None"`.

### [Medium] Challenge 3: PyTorch Loader Crash on Empty Splits
- **Assumption challenged**: That the training, validation, and test splits always contain at least one user with ratings.
- **Attack scenario**: If the dataset contains very few interactions, or if `min_interactions` is adjusted dynamically such that a split is empty (e.g., a user has exactly 2 ratings, leaving train split empty), `train.csv` will be saved as an empty file with only the header row.
- **Blast radius**: When reading the empty CSV, Pandas defaults the columns to `object` dtype. PyTorch throws `TypeError: can't convert np.ndarray of type numpy.object_` during `torch.tensor` creation, causing the entire loader and model initialization to fail.
- **Mitigation**: Add a check in `MovieLensDataLoader` to verify if a split is empty, and initialize tensors with correct dtypes and shapes (e.g., zero-length tensors of type `torch.long`).

### [Low] Challenge 4: Mixed Dtype in zip_code Column (DtypeWarning)
- **Assumption challenged**: That `zip_code` values are uniform strings.
- **Attack scenario**: MovieLens-1M contains some alphanumeric/hyphenated zip codes (e.g. `95014-1234`) and some purely numeric zip codes. Pandas infers purely numeric ones as `int` and alphanumeric ones as `str`.
- **Blast radius**: Triggers a `DtypeWarning` when reading the processed CSVs. Mixed string/integer columns can cause downstream errors in feature engineering or serialization when feeding data into re-ranking GBDT models.
- **Mitigation**: Force `zip_code` dtype to string during raw file reading and processed file loading: `pd.read_csv(..., dtype={'zip_code': str})`.

### [Low] Challenge 5: E2E Mock Preprocessor Divergence
- **Assumption challenged**: That E2E tests using mock data reflect production behavior.
- **Attack scenario**: `tests/e2e/test_f1_data.py::test_empty_ratings_input` checks if the preprocess script exits with non-zero on empty input.
- **Blast radius**: In mock mode, the test fails because `tests/e2e/mock_preprocess.py` does not check for empty dataframes and exits with status 0, leaving empty processed files, whereas `src/data/preprocess.py` throws `ValueError` and exits non-zero.
- **Mitigation**: Align `mock_preprocess.py` error-handling logic with `preprocess.py`.

---

## Stress Test Results

- **Empty raw files** → ValueError (preprocess.py) / Exit 0 (mock_preprocess.py) → **FAIL (for mock preprocess)**
- **Borderline users (< 5 ratings)** → Excluded from train/val/test splits → **PASS**
- **User with exactly 2 ratings (min_interactions=2)** → Empty train split → **FAIL (PyTorch loader crashes on object dtype)**
- **Missing user metadata in `users.dat`** → Left join writes NaNs → **FAIL (PyTorch loader silently casts NaNs to long integer `-9223372036854775808`)**
- **Missing movie metadata in `movies.dat`** → Left join writes NaNs → **FAIL (String fields loaded as floats, causing type mismatches)**
- **Massive/Negative timestamps** → Deterministic chronological sorting → **PASS**
- **Duplicate ratings** → Sorting preserves duplicates in chronological order → **PASS**
- **Production file integrity** → Correct sizes, zero NaNs, no temporal leakage, but mixed `zip_code` types → **PASS (leakage/nulls) / FAIL (zip_code types)**

---

## Unchallenged Areas

- **FAISS Candidate Generation & Re-ranking GBDTs**: Out of scope for Milestone 1 data pipeline verification. Only the input/output boundaries of the data pipeline were reviewed.
