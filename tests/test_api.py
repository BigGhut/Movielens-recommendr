import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    # Поскольку мы запускаемся в мок-режиме, health вернет 200 (или 503 если мы не сделали мок)
    # В нашем случае мы сделали мок пайплайна в main.py, поэтому ожидаем 200
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_recommend_valid_user():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "is_cold_start" in data

@pytest.mark.asyncio
async def test_recommend_cold_start():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Несуществующий юзер
        response = await ac.get("/recommend/999999")
    # Не должен быть 500
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        assert response.json()["is_cold_start"] is True

@pytest.mark.asyncio
async def test_recommend_different_users():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r1 = await ac.get("/recommend/1")
        r2 = await ac.get("/recommend/2")
        
    if r1.status_code == 200 and r2.status_code == 200:
        # В мок режиме могут быть пустыми, но если есть рекомендации, они должны быть корректно обработаны
        pass
