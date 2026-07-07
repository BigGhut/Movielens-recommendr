# Handoff Report - Challenger 1

## 1. Observation

I ran empirical stress tests and production split file checks using custom test suites targeting `src/data/preprocess.py` and `src/data/loader.py`. The following specific failures/warnings were observed:

- **Type conversion warnings on loading production datasets**:
  ```
  Z:\pet-project\recsys-two-tower\tests\test_challenger_m1.py:239: DtypeWarning: Columns (7) have mixed types. Specify dtype option on import or set low_memory=False.
    train_df = pd.read_csv(processed_dir / "train.csv")
  ```
- **Silent Demographics Corruption**: When simulating a user missing from `users.dat` (User 1 in `test_missing_user_metadata_crashes_loader`), the preprocessing merged `NaN` demographic values. The `MovieLensDataLoader` loaded it successfully, but PyTorch cast NaNs silently to:
  ```
  Age tensor: tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808])
  Occupation tensor: tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808])
  ```
- **Type Mismatches on Missing Movie Metadata**: When simulating a movie missing from `movies.dat` (Movie 101 in `test_missing_movie_metadata_causes_string_issues`), the loaded title and genres in `MovieLensDataset` contained float `NaN` values rather than strings.
- **Empty Split Initialization Crash**: When a split (e.g., `train.csv`) was empty (due to sparse history where users had exactly 2 ratings, leaving train empty), PyTorch initialization threw:
  ```
  TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
  ```
- **Mock Prepreprocess Divergence**: Running pytest in mock mode resulted in a failure:
  ```
  FAILED tests/e2e/test_f1_data.py::test_empty_ratings_input - AssertionError: assert 0 != 0
  ```
  This is because `mock_preprocess.py` exited with status 0 on empty input, whereas the test expects it to raise an error and exit non-zero (which the production script does).

---

## 2. Logic Chain

1. **Observed**: `DtypeWarning` for column 7 (`zip_code`) when loading `train.csv`.
   * **Inference**: Alphanumeric zip codes (e.g., `95014-1234`) and numeric zip codes (e.g., `90210`) in MovieLens-1M lead to mixed typing (string & integer) in Pandas. This can crash GBDTs or downstream serialization.
2. **Observed**: Missing user records lead to NaN age/occupation values, which PyTorch casts to `-9223372036854775808` long tensors.
   * **Inference**: PyTorch lacks built-in representation for integer `NaN` and silently converts float `NaN` to garbage minimum `long` integer values. This corrupts inputs during retrieval model training without triggering an exception.
3. **Observed**: Missing movie records result in float `NaN` titles/genres.
   * **Inference**: Title/Genres represent text fields. Having `NaN` floats in these list elements causes crashes in tokenizers or encoders expecting string signatures.
4. **Observed**: Empty split csv yields `object` dtype array.
   * **Inference**: When Pandas reads a CSV with zero rows, it defaults columns to `object` type. PyTorch's `torch.tensor()` cannot convert numpy arrays of type `object` and crashes.
5. **Observed**: `mock_preprocess.py` does not check for empty dataframes.
   * **Inference**: The mock preprocessor doesn't match the production error-handling contract, causing E2E tests checking empty file edge cases to fail during mock runs.

---

## 3. Caveats

- We only analyzed the data pipeline (Milestone 1) scripts (`src/data/preprocess.py`, `src/data/loader.py`).
- We did not investigate model architectures or re-ranking features.
- We assumed the raw MovieLens-1M dataset is the primary dataset and didn't check other versions.

---

## 4. Conclusion

The data pipeline has correct temporal splitting behavior and ensures no data leaks under normal circumstances. However, it is vulnerable to silent data corruption (garbage user tensors on missing users), type crashes (float NaNs in string movie attributes), startup crashes (empty split dataframes), and warnings (mixed zip codes). Additionally, the mock preprocessor needs to be aligned with the production error-handling behavior.

---

## 5. Verification Method

To verify these issues independently:
1. Run the challenger test suite we implemented:
   ```bash
   pytest tests/test_challenger_m1.py -v -s
   ```
   All tests should pass, showing successful detection of each bug:
   - `test_missing_user_metadata_crashes_loader` (detects silent demographics corruption)
   - `test_missing_movie_metadata_causes_string_issues` (detects float NaNs in text attributes)
   - `test_user_with_fewer_than_3_interactions_behavior` (detects empty DataFrame PyTorch casting type crash)
   - `test_production_file_integrity` (detects mixed zip code dtypes and validates temporal splits)
2. Run E2E tests in mock mode to observe the mock preprocess divergence failure:
   ```bash
   pytest tests/e2e/ -v --use-mock-data
   ```
