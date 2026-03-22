# SignalRank

**Industrial-grade retrieval and ranking framework with Python ML pipelines, SQL feature workflows, and C++-accelerated serving components.**

SignalRank is a production-style recommendation system repository that demonstrates how to build and operate a large-scale personalization stack across offline training, online inference, and experimentation.

## Motivation

Modern recommendation stacks for content, ads, and marketplace feeds are multi-stage systems:

1. Candidate generation narrows billions of items to a tractable set.
2. Retrieval uses learned embeddings to find semantically relevant candidates.
3. Ranking predicts engagement/conversion probability with richer features.
4. Reranking applies business constraints and diversity.

SignalRank implements this architecture end-to-end with:
- **Python** for model training, orchestration, and FastAPI serving.
- **SQL** for feature tables, training set construction, diagnostics, and evaluation.
- **C++** for low-latency vector retrieval primitives in the online path.

## Final architecture

```mermaid
flowchart LR
    A[Raw Events / Metadata] --> B[SQL Feature Pipelines]
    B --> C[Feature Store Tables]
    A --> D[LLM Content Enrichment]
    D --> C

    C --> E[Two-Tower Retrieval Training]
    C --> F[Ranker Training]
    E --> G[Item Embedding Index]

    G --> H[C++ ANN Top-K Service Module]
    H --> I[Python Retrieval Service]
    I --> J[Ranker Scoring]
    J --> K[Reranker Constraints & Diversity]
    K --> L[/recommend API Response]

    C --> M[Offline Evaluation SQL]
    E --> M
    F --> M
```

## Repository layout

```text
SignalRank/
├── README.md
├── pyproject.toml
├── Makefile
├── docs/
│   └── architecture.md
├── configs/
│   ├── retrieval.yaml
│   ├── ranking.yaml
│   └── serving.yaml
├── cpp/
│   └── ann_index.cpp
├── src/
│   ├── signalrank/
│   │   ├── common/
│   │   ├── data/
│   │   ├── features/
│   │   ├── llm/
│   │   ├── retrieval/
│   │   ├── ranking/
│   │   ├── rerank/
│   │   ├── evaluation/
│   │   ├── orchestration/
│   │   └── serving/
│   └── signalrank_cpp/
│       └── __init__.py
├── sql/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   └── diagnostics/
├── scripts/
│   ├── run_feature_sql.py
│   ├── train_retrieval.py
│   ├── train_ranker.py
│   ├── evaluate_offline.py
│   └── run_server.py
├── tests/
│   ├── test_metrics.py
│   └── test_pipeline_smoke.py
└── data/sample/
    ├── interactions.csv
    ├── users.csv
    └── items.csv
```

## Why Python + SQL + C++ are all necessary

- **Python** is used for iterative ML work (PyTorch modeling, feature joins, orchestration, online API composition).
- **SQL** is the source of truth for reproducible feature definitions, training labels, data quality diagnostics, and evaluation slices.
- **C++** addresses retrieval bottlenecks in online serving by providing low-overhead vector scoring and top-k selection under latency budgets.

## Implementation plan

1. Build feature datasets with SQL and materialize model-ready views.
2. Train two-tower retrieval model and export item/user embeddings.
3. Build C++ ANN top-k extension and integrate in retrieval service.
4. Train ranker (Wide+Deep style MLP with dense + categorical embeddings).
5. Compose retrieval + ranking + reranking in FastAPI endpoints.
6. Run offline metrics (AUC, log loss, NDCG@K, MRR@K) and diagnostics.

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) Build C++ extension

```bash
python -m pip install -e .
```

### 3) Run feature + training pipeline

```bash
python scripts/run_feature_sql.py
python scripts/train_retrieval.py
python scripts/train_ranker.py
python scripts/evaluate_offline.py
```

### 4) Start API

```bash
python scripts/run_server.py
```

Then test:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 101, "context": {"hour": 18, "surface": "home"}}'
```

## Online API endpoints

- `POST /retrieve`: user/context -> embedding retrieval candidates.
- `POST /rank`: candidates + feature context -> ranking scores.
- `POST /recommend`: full pipeline including reranking.

Latency-aware design notes:
- Precomputed item embeddings + C++ top-k scoring for retrieval.
- Lightweight Python orchestration only in request path.
- Avoid per-request heavyweight joins by serving compact feature snapshots.

## Large-scale system mapping

This repo maps directly to high-scale personalization design:
- Candidate generation via interaction priors and taxonomy filters.
- Learned retrieval via two-tower representation learning.
- Feature-rich ranking with calibrated probabilities.
- Reranking for diversity/freshness policy constraints.
- Offline SQL metrics and diagnostics for experiment readiness.
- API composition pattern suitable for A/B routing and shadow mode.

## Future improvements

- ANN graph index (HNSW/IVF) with incremental updates.
- Real streaming feature store integration.
- Counterfactual evaluation and IPS/DR estimators.
- Multi-objective ranker (engagement + revenue + quality).

## License

MIT (see `LICENSE`).
