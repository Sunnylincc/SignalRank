# Evaluation Guide

## Offline metrics

Python:
- AUC
- Log Loss
- NDCG@K
- MRR@K

SQL:
- Surface/country metric slices
- Accuracy@0.5
- Feature diagnostics (null rate and quantiles)
- Category coverage

## Workflow

1. Run feature + training SQL:
   ```bash
   python scripts/run_feature_sql.py
   ```
2. Run evaluation script:
   ```bash
   python scripts/evaluate_offline.py
   ```

## Why SQL + Python metrics together
- SQL provides reproducible table-level analysis and slice monitoring.
- Python metrics mirror model-level evaluation used in training loops.
