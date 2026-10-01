import pandas as pd
from pathlib import Path

def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Загрузка рейтингов."""
    file_path = data_dir / "ratings.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )

def load_movies(data_dir: Path) -> pd.DataFrame:
    """Загрузка фильмов."""
    file_path = data_dir / "movies.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1"
    )

def load_users(data_dir: Path) -> pd.DataFrame:
    """Загрузка пользователей."""
    file_path = data_dir / "users.dat"
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1"
    )

def temporal_split(ratings: pd.DataFrame, min_ratings: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Выполняет строгое темпоральное разделение данных:
    последнее взаимодействие пользователя -> test,
    предпоследнее -> validation,
    остальные -> train.
    Фильтрует пользователей с < min_ratings.
    """
    # Фильтрация
    user_counts = ratings.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered_ratings = ratings[ratings["user_id"].isin(valid_users)].copy()
    
    # Сортировка по времени
    filtered_ratings.sort_values(["user_id", "timestamp"], inplace=True)
    
    # Разделение
    test_df = filtered_ratings.groupby("user_id").tail(1)
    
    # Исключаем тестовые записи и берем последние из оставшихся
    rest_df = filtered_ratings.drop(test_df.index)
    val_df = rest_df.groupby("user_id").tail(1)
    
    # Все что осталось идет в train
    train_df = rest_df.drop(val_df.index)
    
    return train_df, val_df, test_df

def create_id_mappings(ratings: pd.DataFrame) -> tuple[dict, dict]:
    """
    Создает словари для перевода реальных user_id и item_id в непрерывные индексы (начиная с 0).
    """
    unique_users = sorted(ratings["user_id"].unique())
    unique_items = sorted(ratings["item_id"].unique())
    
    user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item2idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    return user2idx, item2idx

if __name__ == '__main__':
    data_dir = Path("data/raw/ml-1m")
    if data_dir.exists():
        ratings = load_ratings(data_dir)
        print(f"Загружено {len(ratings)} рейтингов.")
        train, val, test = temporal_split(ratings)
        print(f"Размеры сплитов: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        u2i, i2i = create_id_mappings(train)
        print(f"Уникальных пользователей: {len(u2i)}, уникальных фильмов: {len(i2i)}")
    else:
        print(f"Папка {data_dir} не найдена. Запустите download.py.")
