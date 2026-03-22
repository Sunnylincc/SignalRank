# Serving Guide

## Endpoints

### `POST /retrieve`
Returns retrieval-stage candidates with `retrieval_score`.

### `POST /rank`
Accepts a user + explicit candidate IDs and returns rank-scored items.

### `POST /recommend`
Runs full pipeline: candidate generation -> retrieval -> ranking -> reranking.

## Request shape

```json
{
  "user_id": 101,
  "context": {"hour": 18, "surface": "home", "country": "US"},
  "top_k": 10
}
```

## Response shape

```json
{
  "user_id": 101,
  "items": [
    {
      "item_id": 1001,
      "retrieval_score": 0.42,
      "rank_score": 0.36,
      "final_score": 0.35,
      "taxonomy": "electronics"
    }
  ]
}
```

## Latency notes
- Candidate generation bounds search space.
- Retrieval top-k runs in C++ (`signalrank_cpp`) when extension is built.
- Python layer orchestrates stage composition and schema handling.
