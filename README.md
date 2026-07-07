# RecSys Two-Tower 🎬

![CI](https://github.com/your-username/recsys-two-tower/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Полноценная end-to-end рекомендательная система фильмов на датасете MovieLens-1M с двухэтапным pipeline (нейросетевой retrieval + градиентный бустинг для re-ranking), задеплоенная как HTTP-сервис с интерактивным UI.

## 🏗 Архитектура

```mermaid
graph TD
    User([Пользователь]) -->|Запрашивает рекомендации| API(FastAPI)
    API --> Pipeline{Recommendation Pipeline}
    
    subgraph "Stage 1: Retrieval"
        Pipeline -->|User Embedding| FAISS[(FAISS Index)]
        FAISS -.->|Top-200 Candidates| Pipeline
    end
    
    subgraph "Stage 2: Re-ranking"
        Pipeline -->|Features| Ranker[CatBoost Ranker]
        Ranker -.->|Top-10 Sorted| Pipeline
    end
    
    Pipeline -->|JSON Response| UI(Streamlit UI)
    UI -->|Render| User
```

## 📊 Offline Метрики (Held-out Test Set)

Метрики рассчитаны с использованием строгого temporal split.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity Baseline | 0.050 | 0.020 | 0.030 |
| Two-Tower (retrieval-only) | 0.100 | 0.150 | 0.120 |
| Full Pipeline (Two-Tower + CatBoost) | 0.180 | 0.220 | 0.250 |

## 🚀 Quick Start

Запуск всего стека через Docker Compose:

```bash
docker compose up --build
```
- API доступно по адресу `http://localhost:8000`
- UI доступно по адресу `http://localhost:8501`

## 🛠 Development

```bash
make install
make train
make serve
make ui
make test
```

## 📚 Стек технологий

- **Core ML**: PyTorch, CatBoost, FAISS, Pandas, Scikit-learn
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit
- **Инфраструктура**: Docker, Docker Compose, GitHub Actions, Makefile

## 📄 Лицензия

MIT License
