# RecSys Two-Tower

![CI](https://github.com/BigGhut/Movielens-recommendr/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Двухэтапные рекомендации фильмов на MovieLens-1M: two-tower retrieval в FAISS и CatBoost, который переставляет топ-200. Сервис — FastAPI, интерфейс — Vite.

## Архитектура

Пользователь запрашивает рекомендации у FastAPI. Пайплайн считает вектор пользователя, достаёт из FAISS 200 ближайших фильмов, которых нет в истории, и CatBoost оставляет топ-10.

Башни симметричные, на выходе L2-норма, индекс — FAISS inner product.

- Башня фильма: `item_id`, multi-hot жанров и год из названия.
- Башня пользователя: `user_id`, средние жанр и год последних 10 фильмов из train и среднее их id-эмбеддингов.
- Обучение: in-batch InfoNCE и 8 hard negatives, выбранных пропорционально популярности.
- Ранкер: CatBoost YetiRank. Итоговый скор — порядок FAISS плюс остаток модели. Вес остатка подбирается на отложенных пользователях. Отдельный признак `recent_neighbor_score` — косинус кандидата со средним вектором последних фильмов.

GNN и мультимодальные энкодеры в репозитории есть, но в этот индекс не пишут.

## Offline-метрики

Сплит — leave-last-out внутри пользователя, не общий срез по времени: в train одного пользователя могут лежать события позже теста другого. Последнее взаимодействие — test, предпоследнее — validation, остальное — train. Пользователи с менее чем 5 оценками отброшены.

После этого остаётся 6040 пользователей, у каждого одно тестовое событие. В среднее входят 3563, у кого эта оценка не ниже 4. Остальные 2477 из метрики выкинуты, а не записаны нулём: иначе каждое число в таблице умножилось бы на 3563/6040. У пользователя в оценке один позитив, поэтому Precision@10 остаётся маленьким.

Таблица округлена из [reports/metrics.json](reports/metrics.json). Это прогон `evaluate.py` на коде `79982d0`, seed 42 у two-tower и у CatBoost. Локальный `metrics.json` в git не входит.

| Model | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.0022 | 0.0222 | 0.0104 | 0.0070 |
| Two-Tower (retrieval-only) | 0.0096 | 0.0957 | 0.0430 | 0.0274 |
| Full Pipeline (Two-Tower + CatBoost) | 0.0166 | 0.1659 | 0.0867 | 0.0627 |

## Запуск

Нужны Python 3.10+ и Node.js. Датасет и веса в репозиторий не входят.

```bash
pip install -e ".[dev]"
python -m src.data.download
make train
make serve-api
make serve-ui
```

`make train` учит two-tower, затем CatBoost и перезаписывает `metrics.json`. API поднимается на `http://localhost:8000`, UI Vite — на `http://localhost:5173` и ходит в API за `http://localhost:8000`.

Полезные ручки: `GET /health`, `GET /recommend/{user_id}`, `GET /history/{user_id}`, `GET /metrics`.

## Разработка

```bash
make install
make test
make lint
```

`make install` ставит Python-пакет и зависимости UI. `make serve-api` и `make serve-ui` — отдельные цели.

## Стек

- PyTorch, CatBoost, FAISS, Pandas, scikit-learn
- FastAPI, Pydantic, Uvicorn
- Vite
- GitHub Actions, Makefile

## Лицензия

MIT
