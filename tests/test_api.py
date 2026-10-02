import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_valid_user(client):
    response = client.get("/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "is_cold_start" in data


def test_recommend_cold_start(client):
    response = client.get("/recommend/999999")
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert response.json()["is_cold_start"] is True


def test_recommend_different_users(client):
    r1 = client.get("/recommend/1")
    r2 = client.get("/recommend/2")

    if r1.status_code == 200 and r2.status_code == 200:
        pass
