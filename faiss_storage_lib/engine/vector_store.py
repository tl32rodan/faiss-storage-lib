from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import numpy as np

from faiss_storage_lib.core.schema import VectorDocument

if TYPE_CHECKING:
    import faiss


_SCORE_NORMALIZERS: Dict[int, Callable[[float], float]] = {
    1: lambda d: 1.0 / (1.0 + d),  # METRIC_L2: squared-L2 distance
    0: lambda d: max(0.0, min(1.0, d)),  # METRIC_INNER_PRODUCT: already similarity-like
}


class FaissVectorStore:
    """Manages a FAISS index: add, remove, search, reconstruct, rebuild, and persist."""

    def __init__(self, index_dir: Path, dimension: int) -> None:
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._index_path = index_dir / "faiss.index"
        self._index = self._load_or_create()

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._index.add_with_ids(vectors, ids)

    def remove(self, ids: List[int]) -> None:
        selector = self._faiss.IDSelectorBatch(self._prepare_ids(ids))
        self._index.remove_ids(selector)

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self._index.search(query, top_k)
        normalizer = _SCORE_NORMALIZERS.get(self._metric_type())
        if normalizer is None:
            raise ValueError(
                f"No score normalizer registered for metric_type={self._metric_type()}"
            )
        scores = np.array(
            [[normalizer(float(d)) for d in row] for row in distances],
            dtype="float32",
        )
        return scores, indices

    def reconstruct(self, int_id: int) -> List[float]:
        return self._reconstruct_from(self._index, int_id)

    def rebuild(self, overrides: Dict[str, VectorDocument], uid_int_ids: Dict[str, int]) -> None:
        old_index = self._index
        self._index = self._create()
        vectors: List[List[float]] = []
        ids: List[int] = []
        for uid, int_id in uid_int_ids.items():
            if uid in overrides:
                vector = overrides[uid].vector
            else:
                vector = self._reconstruct_from(old_index, int_id)
                if not vector:
                    continue
            vectors.append(vector)
            ids.append(int_id)
        if vectors:
            self._index.add_with_ids(self._prepare_vectors(vectors), self._prepare_ids(ids))

    def persist(self, path: str) -> None:
        self._faiss.write_index(self._index, path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _metric_type(self) -> int:
        """Return the FAISS metric type of the underlying index."""
        index = self._index
        # IndexIDMap wraps a sub-index; read metric_type from the inner index.
        if hasattr(index, "index"):
            return index.index.metric_type
        return index.metric_type

    def _load_or_create(self) -> "faiss.Index":
        if self._index_path.exists():
            return self._faiss.read_index(str(self._index_path))
        return self._create()

    def _create(self) -> "faiss.Index":
        base_index = self._faiss.IndexFlatL2(self._dimension)
        return self._faiss.IndexIDMap(base_index)

    def _prepare_vectors(self, values: List[List[float]]) -> np.ndarray:
        return np.array(values, dtype="float32")

    def _prepare_ids(self, values: List[int]) -> np.ndarray:
        return np.array(values, dtype="int64")

    def _reconstruct_from(self, index: "faiss.Index", int_id: int) -> List[float]:
        try:
            if hasattr(index, "id_map") and hasattr(index, "index"):
                id_map = self._faiss.vector_to_array(index.id_map)
                matches = np.where(id_map == int_id)[0]
                if matches.size == 0:
                    return []
                return index.index.reconstruct(int(matches[0])).tolist()
            return index.reconstruct(int_id).tolist()
        except RuntimeError:
            return []
