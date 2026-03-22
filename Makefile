.PHONY: install test lint train serve

install:
	pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src tests scripts

train:
	python scripts/run_feature_sql.py && python scripts/train_retrieval.py && python scripts/train_ranker.py

serve:
	python scripts/run_server.py
