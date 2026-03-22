# SignalRank Architecture Deep Dive

## System goals
- Retrieve relevant candidates under strict latency budgets.
- Rank candidates with richer feature interactions.
- Keep feature engineering reproducible through SQL.
- Support experiment loops with robust offline diagnostics.

## Pipeline stages
1. **Data + features (SQL):** User/item/context features are materialized with freshness checks and sparsity diagnostics.
2. **LLM enrichment:** Item titles/descriptions receive semantic tags, taxonomy nodes, and embeddings.
3. **Retrieval training:** Two-tower model learns user/item embedding space with in-batch negatives.
4. **Retrieval serving:** C++ top-k module computes fast dot-product scores over candidate vectors.
5. **Ranking:** Wide+Deep-style neural ranker scores retrieved candidates using dense + categorical features.
6. **Reranking:** Post-processing enforces diversity and freshness constraints.

## Why C++ in retrieval
The dominant bottleneck in naive Python retrieval is repeated dot-product and top-k selection across many vectors for each request. The C++ extension eliminates Python loop overhead and performs contiguous-memory scoring and partial-sort in native code.

## Experimentation model
- Offline metrics: AUC, log loss, NDCG@K, MRR@K.
- Slice-level SQL metrics by surface, country, freshness, and semantic bucket.
- A/B-ready serving interfaces with deterministic model/config snapshots.
