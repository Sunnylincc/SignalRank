from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import signalrank_cpp
except ImportError:
    signalrank_cpp = None


class ANNIndex:
    """Dense vector index with optional C++ acceleration for top-k retrieval."""

    def __init__(self, item_ids: Sequence[int], vectors: np.ndarray):
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.id_to_row = {int(item_id): i for i, item_id in enumerate(self.item_ids.tolist())}

        if self.vectors.shape[0] != len(self.item_ids):
            raise ValueError("vectors and item_ids length mismatch")

        self._index = (
            signalrank_cpp.ANNIndex(self.vectors.tolist(), self.item_ids.tolist()) if signalrank_cpp is not None else None
        )

    def top_k(self, user_vector: np.ndarray, k: int, candidate_ids: Sequence[int] | None = None) -> list[tuple[int, float]]:
        vec = np.asarray(user_vector, dtype=np.float32)
        if candidate_ids is None:
            if self._index is not None:
                return self._index.top_k(vec.tolist(), k)
            scores = self.vectors @ vec
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
            top = top[np.argsort(-scores[top])]
            return [(int(self.item_ids[i]), float(scores[i])) for i in top]

        candidate_ids = [int(i) for i in candidate_ids if int(i) in self.id_to_row]
        if not candidate_ids:
            return []

        if self._index is not None:
            return self._index.top_k_subset(vec.tolist(), candidate_ids, k)

        row_ids = np.array([self.id_to_row[i] for i in candidate_ids], dtype=np.int64)
        sub_vectors = self.vectors[row_ids]
        scores = sub_vectors @ vec
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(candidate_ids[i]), float(scores[i])) for i in top]
