from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import signalrank_cpp
except ImportError:  # fallback for environments without compiled extension
    signalrank_cpp = None


class ANNIndex:
    def __init__(self, item_ids: Sequence[int], vectors: np.ndarray):
        self.item_ids = np.array(item_ids, dtype=np.int64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if signalrank_cpp is not None:
            self._index = signalrank_cpp.ANNIndex(self.vectors.tolist(), self.item_ids.tolist())
        else:
            self._index = None

    def top_k(self, user_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        vec = np.asarray(user_vector, dtype=np.float32)
        if self._index is not None:
            return self._index.top_k(vec.tolist(), k)
        scores = self.vectors @ vec
        top = np.argsort(-scores)[:k]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top]
