import argparse
from pathlib import Path

import pandas as pd


def generate_mock_raw_data(raw_dir: Path):
    """Generate tiny raw MovieLens-1M files for mock training/testing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ratings: user_id::movie_id::rating::timestamp
    # User 1 has 6 ratings -> Should keep (6 >= 5). Train: 4, Val: 1, Test: 1.
    # User 2 has 5 ratings -> Borderline (5 >= 5). Train: 3, Val: 1, Test: 1.
    # User 3 has 4 ratings -> Should filter out (< 5).
    # User 4 has 7 ratings -> Should keep. Train: 5, Val: 1, Test: 1.
    # User 5 has 5 ratings -> All ratings have identical timestamps to test boundary case.
    ratings_data = [
        # User 1 (6 ratings)
        "1::101::4::1000",
        "1::102::5::1001",
        "1::103::3::1002",
        "1::104::4::1003",
        "1::105::5::1004",
        "1::106::2::1005",
        # User 2 (5 ratings - borderline)
        "2::101::3::2000",
        "2::102::4::2001",
        "2::103::5::2002",
        "2::104::2::2003",
        "2::105::4::2004",
        # User 3 (4 ratings - filtered out)
        "3::101::5::3000",
        "3::102::4::3001",
        "3::103::3::3002",
        "3::104::2::3003",
        # User 4 (7 ratings)
        "4::101::5::4000",
        "4::102::4::4001",
        "4::103::3::4002",
        "4::104::5::4003",
        "4::105::4::4004",
        "4::106::3::4005",
        "4::107::5::4006",
        # User 5 (5 ratings - identical timestamps)
        "5::101::5::5000",
        "5::102::4::5000",
        "5::103::3::5000",
        "5::104::5::5000",
        "5::105::2::5000",
    ]
    
    # 2. Users: user_id::gender::age::occupation::zip_code
    users_data = [
        "1::F::1::10::48067",
        "2::M::56::16::70072",
        "3::M::25::15::55117",
        "4::M::45::7::02460",
        "5::F::25::20::94103",
    ]
    
    # 3. Movies: movie_id::title::genres
    movies_data = [
        "101::Toy Story (1995)::Animation|Children's|Comedy",
        "102::Jumanji (1995)::Adventure|Children's|Fantasy",
        "103::Grumpier Old Men (1995)::Comedy|Romance",
        "104::Waiting to Exhale (1995)::Comedy|Drama",
        "105::Father of the Bride Part II (1995)::Comedy",
        "106::Heat (1995)::Action|Crime|Thriller",
        "107::Sabrina (1995)::Comedy|Romance",
    ]
    
    with open(raw_dir / "ratings.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(ratings_data) + "\n")
        
    with open(raw_dir / "users.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(users_data) + "\n")
        
    with open(raw_dir / "movies.dat", "w", encoding="latin-1") as f:
        f.write("\n".join(movies_data) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/ml-1m")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    # If raw files don't exist, generate them. An existing empty file is an error,
    # same as src.data.preprocessing.load_raw_data.
    if not (raw_dir / "ratings.dat").exists():
        generate_mock_raw_data(raw_dir)
    for name in ("ratings.dat", "users.dat", "movies.dat"):
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Raw data file is empty: {path}")
        
    # Read the raw files
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1"
    )
    
    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    
    # 1. Filter out users with < 5 interactions
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()
    
    # 2. Chronological temporal split per user
    # To handle identical timestamps deterministically, sort by [user_id, timestamp, movie_id]
    sorted_ratings = filtered_ratings.sort_values(
        by=["user_id", "timestamp", "movie_id"]
    ).reset_index(drop=True)
    
    grouped = sorted_ratings.groupby("user_id")
    group_size = grouped["user_id"].transform("count")
    cum_count = grouped.cumcount()
    
    train_mask = cum_count < (group_size - 2)
    val_mask = cum_count == (group_size - 2)
    test_mask = cum_count == (group_size - 1)
    
    train = sorted_ratings[train_mask].copy()
    val = sorted_ratings[val_mask].copy()
    test = sorted_ratings[test_mask].copy()
    
    # 3. Merge metadata
    def merge_metadata(df):
        df = df.merge(users, on="user_id", how="left")
        df = df.merge(movies, on="movie_id", how="left")
        return df
        
    train_merged = merge_metadata(train)
    val_merged = merge_metadata(val)
    test_merged = merge_metadata(test)
    
    # Save to CSV
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_merged.to_csv(processed_dir / "train.csv", index=False)
    val_merged.to_csv(processed_dir / "val.csv", index=False)
    test_merged.to_csv(processed_dir / "test.csv", index=False)
    
    print(f"Mock preprocessing complete. Saved train, val, test CSVs in {processed_dir}")

if __name__ == "__main__":
    main()
