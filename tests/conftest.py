import numpy as np
import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use-mock-data",
        action="store_true",
        default=False,
        help="Use mock/synthetic data for the e2e suite. MovieLens files are not in the repo.",
    )


@pytest.fixture
def synthetic_ratings():
    # Создадим 10 юзеров, у каждого по 6 оценок (чтобы пройти фильтр min_ratings=5)
    data = []
    timestamp = 1000
    for u in range(1, 11):
        for i in range(1, 7):
            data.append([u, i, np.random.randint(1, 6), timestamp])
            timestamp += 10 # чтобы был строгий порядок по времени
            
    # Добавим одного юзера с 3 оценками (должен отфильтроваться)
    for i in range(1, 4):
        data.append([11, i, 5, timestamp])
        timestamp += 10
        
    df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])
    return df

@pytest.fixture
def synthetic_movies():
    data = []
    for i in range(1, 20):
        data.append([i, f"Movie {i} (199{i%10})", "Action|Drama" if i%2==0 else "Comedy"])
    df = pd.DataFrame(data, columns=["item_id", "title", "genres"])
    return df

@pytest.fixture
def synthetic_users():
    data = []
    for u in range(1, 12):
        data.append([u, "M" if u%2==0 else "F", 25, 1, "12345"])
    df = pd.DataFrame(data, columns=["user_id", "gender", "age", "occupation", "zip"])
    return df
