.PHONY: install test lint features train-retrieval train-ranker evaluate train-all serve

install:
	pip install -e .[dev]

features:
	python scripts/run_feature_sql.py

train-retrieval:
	python scripts/train_retrieval.py

train-ranker:
	python scripts/train_ranker.py

evaluate:
	python scripts/evaluate_offline.py

train-all: features train-retrieval train-ranker evaluate

serve:
	python scripts/run_server.py

test:
	pytest -q

lint:
	ruff check src tests scripts
