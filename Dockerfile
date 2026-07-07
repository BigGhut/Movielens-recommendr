FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Установка зависимостей проекта
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY app/ app/
COPY metrics.json metrics.json

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
