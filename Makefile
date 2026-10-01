.PHONY: install train serve-api serve-ui test lint all

install:
	pip install -e ".[dev]"
	cd app && npm install

train:
	python -m src.models.train_retrieval
	python -m src.models.train_ranker
	python evaluate.py

serve-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-ui:
	cd app && npm run dev

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

all: install train test
