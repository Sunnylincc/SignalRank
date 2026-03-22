# Development Guide

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Common commands
```bash
python scripts/run_feature_sql.py
python scripts/train_retrieval.py
python scripts/train_ranker.py
python scripts/evaluate_offline.py
python scripts/run_server.py
pytest -q
```

## Code quality expectations
- Keep module boundaries explicit across retrieval, ranking, reranking, and serving.
- Preserve SQL readability and table naming consistency in `mart` schema.
- Keep C++ extension focused on performance-critical kernels only.
