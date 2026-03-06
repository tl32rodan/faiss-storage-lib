from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from faiss_storage_lib.core.schema import VectorDocument
from faiss_storage_lib.engine.index_config import IndexConfig, IndexType

if TYPE_CHECKING:
    import faiss

# IVF indexes require at least this many training vectors per cluster.
_IVF_MIN_TRAIN_FACTOR = 39


class FaissVectorStore:
    """Manages a FAISS index: add, remove, search, reconstruct, rebuild, and persist.

    Supports four index strategies via IndexConfig:
      - FLAT    : IndexFlatL2 + IndexIDMap (exact, no training)
      - IVF_FLAT: IndexIVFFlat (approximate, training required)
      - IVF_SQ  : IndexIVFScalarQuantizer (compressed, training required)
      - IVF_PQ  : IndexIVFPQ (highly compressed, training required;
                  exact vectors fetched from doc store for reconstruction)

    For IVF_PQ, a fetch_vector_blob callback must be supplied so that
    _reconstruct_from() can return exact (not quantised) vectors during
    rebuild() and reconstruct() calls.
    """

    def __init__(
        self,
        index_dir: Path,
        dimension: int,
        index_config: Optional[IndexConfig] = None,
        *,
        fetch_vector_blob: Optional[Callable[[int], List[float]]] = None,
    ) -> None:
        import faiss

        self._faiss = faiss
        self._dimension = dimension
        self._config = index_config or IndexConfig()
        self._index_path = index_dir / "faiss.index"
        # Callback for IVF_PQ exact-vector reconstruction from SQLite.
        self._fetch_vector_blob = fetch_vector_blob
        self._index = self._load_or_create()

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._ensure_trained(vectors)
        self._index.add_with_ids(vectors, ids)

    def remove(self, ids: List[int]) -> None:
        selector = self._faiss.IDSelectorBatch(self._prepare_ids(ids))
        self._index.remove_ids(selector)

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self._config.nprobe
        return self._index.search(query, top_k)

    def reconstruct(self, int_id: int) -> List[float]:
        return self._reconstruct_from(self._index, int_id)

    def rebuild(self, overrides: Dict[str, VectorDocument], uid_int_ids: Dict[str, int]) -> None:
        old_index = self._index
        # For already-trained IVF indexes: clone the trained structure and reset
        # only the inverted lists.  This preserves quantizer centroids and PQ
        # sub-quantizers, avoiding a retrain that would fail when the remaining
        # vector count falls below the training threshold (e.g. after a delete).
        if old_index.is_trained and self._config.index_type != IndexType.FLAT:
            self._index = self._faiss.clone_index(old_index)
            self._index.reset()
            self._index.set_direct_map_type(self._faiss.DirectMap.Hashtable)
        else:
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
            vecs_arr = self._prepare_vectors(vectors)
            ids_arr = self._prepare_ids(ids)
            self._ensure_trained(vecs_arr)  # no-op when index is already trained
            self._index.add_with_ids(vecs_arr, ids_arr)

    def persist(self, path: str) -> None:
        self._faiss.write_index(self._index, path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_create(self) -> "faiss.Index":
        if self._index_path.exists():
            idx = self._faiss.read_index(str(self._index_path))
            if hasattr(idx, "nprobe"):
                idx.nprobe = self._config.nprobe
            # Ensure reconstruct() works on IVF indexes loaded from disk.
            if hasattr(idx, "set_direct_map_type") and not hasattr(idx, "id_map"):
                idx.set_direct_map_type(self._faiss.DirectMap.Hashtable)
            return idx
        return self._create()

    def _create(self) -> "faiss.Index":
        cfg = self._config
        d = self._dimension
        if cfg.index_type == IndexType.FLAT:
            return self._faiss.IndexIDMap(self._faiss.IndexFlatL2(d))
        quantizer = self._faiss.IndexFlatL2(d)
        if cfg.index_type == IndexType.IVF_FLAT:
            idx = self._faiss.IndexIVFFlat(quantizer, d, cfg.nlist)
        elif cfg.index_type == IndexType.IVF_SQ:
            sq = getattr(self._faiss.ScalarQuantizer, cfg.sq_type)
            idx = self._faiss.IndexIVFScalarQuantizer(quantizer, d, cfg.nlist, sq)
        elif cfg.index_type == IndexType.IVF_PQ:
            idx = self._faiss.IndexIVFPQ(quantizer, d, cfg.nlist, cfg.pq_m, cfg.pq_nbits)
        else:
            raise ValueError(f"Unknown IndexType: {cfg.index_type}")
        idx.nprobe = cfg.nprobe
        # Enable reconstruct(id) on IVF indexes.  Without a Hashtable direct
        # map, reconstruct() raises "direct map not initialized" and Array mode
        # conflicts with add_with_ids (arbitrary non-sequential IDs).
        idx.set_direct_map_type(self._faiss.DirectMap.Hashtable)
        return idx

    def _ensure_trained(self, vectors: np.ndarray) -> None:
        """Train the index if it has not been trained yet (IVF only).

        Collects all vectors from the current index plus the incoming batch
        as training data. Raises RuntimeError if there are not enough vectors
        to meet FAISS's minimum training requirement (39 × nlist).
        """
        if self._index.is_trained:
            return
        min_train = _IVF_MIN_TRAIN_FACTOR * self._config.nlist
        # IVF_PQ: each sub-quantizer has 2^pq_nbits centroids; FAISS requires
        # n >= 2^pq_nbits (hard limit, not a quality rule-of-thumb).
        if self._config.index_type == IndexType.IVF_PQ:
            min_train = max(min_train, 2**self._config.pq_nbits)
        # Reconstruct any vectors already in the index (e.g. after load from disk
        # of an untrained index, which should not normally happen).
        existing: List[List[float]] = []
        if self._index.ntotal > 0:
            existing = self._collect_existing_vectors(self._index)
        all_vecs = np.vstack([self._prepare_vectors(existing), vectors]) if existing else vectors
        if len(all_vecs) < min_train:
            raise RuntimeError(
                f"IVF index requires at least {min_train} vectors to train "
                f"(got {len(all_vecs)}). "
                f"Either add more vectors first or use IndexType.FLAT for small datasets."
            )
        self._index.train(all_vecs)

    def _collect_existing_vectors(self, index: "faiss.Index") -> List[List[float]]:
        """Reconstruct all vectors currently stored in the given index."""
        total = index.ntotal
        if total == 0:
            return []
        # Retrieve all stored IDs from the index's sa_encode / reconstruct_n path.
        # Use reconstruct_n for a bulk fetch when available.
        try:
            vecs = index.reconstruct_n(0, total)
            return vecs.tolist()
        except Exception:
            return []

    def _prepare_vectors(self, values: List[List[float]]) -> np.ndarray:
        return np.array(values, dtype="float32")

    def _prepare_ids(self, values: List[int]) -> np.ndarray:
        return np.array(values, dtype="int64")

    def _reconstruct_from(self, index: "faiss.Index", int_id: int) -> List[float]:
        """Reconstruct a vector by its user-facing integer ID.

        For IVF_PQ, reconstruction via the FAISS index is lossy (product
        quantisation). When a fetch_vector_blob callback is available, exact
        vectors are retrieved from SQLite instead.

        For IndexIDMap (FLAT), the internal sequential position is resolved
        via the id_map before calling reconstruct().

        For bare IVF_FLAT / IVF_SQ, reconstruct(int_id) works directly.
        """
        # IVF_PQ: prefer exact vector from SQLite blob.
        if self._config.index_type == IndexType.IVF_PQ and self._fetch_vector_blob is not None:
            vec = self._fetch_vector_blob(int_id)
            if vec:
                return vec
        try:
            if hasattr(index, "id_map") and hasattr(index, "index"):
                # IndexIDMap: map user ID → internal sequential position.
                id_map = self._faiss.vector_to_array(index.id_map)
                matches = np.where(id_map == int_id)[0]
                if matches.size == 0:
                    return []
                return index.index.reconstruct(int(matches[0])).tolist()
            # IVF_FLAT / IVF_SQ: reconstruct directly by user ID.
            return index.reconstruct(int_id).tolist()
        except RuntimeError:
            return []
