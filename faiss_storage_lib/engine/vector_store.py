from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from faiss_storage_lib.core.schema import VectorDocument

if TYPE_CHECKING:
    import faiss


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
        return self._index.search(query, top_k)

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
