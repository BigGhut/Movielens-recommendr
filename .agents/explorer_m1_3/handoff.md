# Handoff Report: MovieLens-1M Data Pipeline (Milestone 1)

## 1. Observation
- Checked `z:/pet-project/recsys-two-tower/PROJECT.md` (lines 50-53):
  > ### Data Loader ↔ Retrieval & Re-ranking
  > - **Inputs**: Raw MovieLens-1M files (`ratings.dat`, `users.dat`, `movies.dat`).
  > - **Outputs**: Train, validation, and test pandas DataFrames / CSVs containing `user_id`, `movie_id`, `rating`, `timestamp`, `title`, `genres`, etc.
  > - **Contract**: No rating in test-split may have a timestamp earlier than any rating in train/validation split for the same user.
- Checked `z:/pet-project/recsys-two-tower/ORIGINAL_REQUEST.md` (lines 15-18):
  > ### R1. Data Pipeline с корректным temporal split
  > Реализовать загрузку датасета MovieLens-1M и его обработку. Разбивка на train/validation/test **обязана** быть **temporal** (по времени выставления рейтинга): для каждого пользователя последние взаимодействия уходят в test, предпоследние — в validation, остальные — в train. Random split запрещён — он создаёт data leakage. Фильтровать пользователей с менее чем 5 рейтингами.
- Listed the `.agents` folder, showing no pre-existing code files in `src/` or `tests/`.

## 2. Logic Chain
1. From the requirement in `ORIGINAL_REQUEST.md` (Observation above), users with fewer than 5 ratings must be filtered. This is necessary because the split allocates 1 rating to validation and 1 rating to test. If a user had fewer than 5 ratings, their training set history would contain less than 3 ratings, hindering user representation learning.
2. From the contract in `PROJECT.md` (Observation above), test rating timestamps must not be earlier than any train/validation rating timestamp for the same user.
3. Chronological sorting by `(user_id, timestamp)` allows identifying the chronological order of ratings for each user.
4. To guarantee deterministic partitioning when multiple ratings share the exact same timestamp, a secondary tie-breaker sorting on `movie_id` is introduced.
5. In pandas, grouping by `user_id` and calculating `cumcount(ascending=False)` assigns rank index:
   - Rank `0` represents the latest interaction (assigned to `test`).
   - Rank `1` represents the second-to-latest interaction (assigned to `validation`).
   - Rank `>= 2` represents all earlier interactions (assigned to `train`).
6. Joining the demographic details from `users.dat` and movie information from `movies.dat` prior to saving the files ensures that downstream stages (Retrieval in M2 and Re-ranking in M3) directly receive enriched tables containing the columns: `user_id`, `movie_id`, `rating`, `timestamp`, `gender`, `age`, `occupation`, `zip_code`, `title`, and `genres`. This conforms to the output interface contract.

## 3. Caveats
- **User-Agent restrictions**: GroupLens hosting servers frequently reject python standard library `urllib` user-agents (throwing HTTP 403 Forbidden). The download template solves this by providing a browser-like string in headers.
- **Encoding compatibility**: MovieLens-1M description files contain special ISO-8859-1 (latin-1) characters in movie titles. Reading with UTF-8 will cause parser exceptions. The loaders must specify `encoding="ISO-8859-1"`.
- **Ties in timestamps**: When a user has multiple ratings with the same timestamp, our tie-breaker is `movie_id` ascending. In such scenarios, the temporal order becomes relative to the ID sorting.
- **Environment**: This design assumes standard modern Python dependencies (`pandas`, `numpy`, and PyTorch for the dataset loaders).

## 4. Conclusion
The proposed plan and template files provide a complete, robust, and leakage-free strategy to download and preprocess the MovieLens-1M dataset. All requirement constraints (filtering < 5 interactions, user-level temporal split, metadata enrichment, deterministic tie-breaking) have been satisfied.

## 5. Verification Method
1. Create `src/data/` directory and populate files:
   - `src/data/download.py` (using template from `analysis.md`)
   - `src/data/preprocess.py` (using template from `analysis.md`)
   - `src/data/dataset.py` (using template from `analysis.md`)
2. Create `tests/test_data_pipeline.py` with the unit test code block.
3. Run the unit test using pytest:
   ```powershell
   pytest tests/test_data_pipeline.py
   ```
4. Verify execution of pipeline:
   ```powershell
   python src/data/download.py
   python src/data/preprocess.py
   ```
5. Invalidation conditions:
   - Any test failure in `tests/test_data_pipeline.py`.
   - The existence of overlapping interactions between `train.csv`, `validation.csv`, and `test.csv`.
   - The presence of user ratings in `test.csv` that are earlier than their corresponding ratings in `train.csv` or `validation.csv`.
