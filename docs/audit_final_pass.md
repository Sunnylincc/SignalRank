# Final-pass Audit Summary

## Highest-priority issues found

1. **Serving/ranking path had toy behavior**
   - `POST /rank` used ad-hoc dynamic objects with synthetic scores instead of a coherent ranker interface.
2. **LLM enrichment was disconnected from retrieval/ranking flow**
   - Enrichment module existed but was not integrated into candidate metadata or feature construction.
3. **SQL evaluation used random predictions**
   - `sql/evaluation/offline_metrics.sql` computed metrics from random scores, weakening credibility.
4. **C++ retrieval was useful but limited**
   - Only full-index top-k existed; no filtered retrieval for candidate subsets.
5. **Pipeline boundaries were shallow**
   - Candidate generation stage was not explicit and retrieval/ranking feature contracts were unclear.
6. **Docs needed stronger production framing**
   - README/docs did not clearly separate implemented vs simplified pieces and lacked development/serving/evaluation guides.

## Upgrade plan executed

- Refactor retrieval/ranking/reranking pipeline around typed request/response models.
- Add explicit candidate generation and filtered ANN retrieval path.
- Integrate LLM enrichment outputs into serving metadata and reranking diversity behavior.
- Replace random SQL evaluation scoring with deterministic baseline model scores derived from real features.
- Extend C++ extension with subset top-k API for latency-aware filtered candidate retrieval.
- Harden docs: architecture, serving, evaluation, development, and end-to-end runbook.
