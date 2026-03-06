"""Tests for IVF-based FAISS index types: IVF_FLAT, IVF_SQ, IVF_PQ.

Each test is parameterised over all three IVF variants. A small nlist=4 is
used so that training only requires 39*4 = 156 vectors, keeping tests fast.
The dimension is chosen to be divisible by the default pq_m=8.
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from faiss_storage_lib import FaissEngine, IndexConfig, IndexType, VectorDocument

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

DIM = 16  # divisible by pq_m=8
NLIST = 4
MIN_TRAIN = 39 * NLIST  # 156

IVF_TYPES = [IndexType.IVF_FLAT, IndexType.IVF_SQ, IndexType.IVF_PQ]


def _make_config(index_type: IndexType) -> IndexConfig:
    # pq_nbits=4 → 2^4=16 PQ centroids; training threshold = max(39*4, 16) = 156.
    return IndexConfig(index_type=index_type, nlist=NLIST, nprobe=NLIST, pq_m=8, pq_nbits=4)


def _random_doc(uid: str, seed: int | None = None) -> VectorDocument:
    rng = np.random.default_rng(seed)
    return VectorDocument(uid=uid, vector=rng.random(DIM).tolist(), payload={"uid": uid})


def _make_docs(n: int, prefix: str = "doc") -> List[VectorDocument]:
    return [_random_doc(f"{prefix}-{i}", seed=i) for i in range(n)]


def _make_engine(tmp_path: str, index_type: IndexType) -> FaissEngine:
    return FaissEngine(tmp_path, DIM, _make_config(index_type))


# -----------------------------------------------------------------------
# Training threshold tests (IVF-specific)
# -----------------------------------------------------------------------


@pytest.mark.parametrize("index_type", IVF_TYPES)
def test_ivf_raises_when_not_enough_vectors_to_train(index_type: IndexType) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, index_type)
        few_docs = _make_docs(MIN_TRAIN - 1)  # one short
        with pytest.raises(RuntimeError, match="IVF index requires"):
            engine.add(few_docs)


@pytest.mark.parametrize("index_type", IVF_TYPES)
def test_ivf_trains_automatically_when_threshold_met(index_type: IndexType) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, index_type)
        docs = _make_docs(MIN_TRAIN)
        engine.add(docs)  # should not raise
        assert engine._vector_store.ntotal == MIN_TRAIN


# -----------------------------------------------------------------------
# Full CRUD lifecycle
# -----------------------------------------------------------------------


@pytest.mark.parametrize("index_type", IVF_TYPES)
def test_ivf_crud_full_lifecycle(index_type: IndexType) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, index_type)
        docs = _make_docs(MIN_TRAIN)

        # --- CREATE ---
        engine.add(docs)
        assert engine._vector_store.ntotal == MIN_TRAIN

        # --- SEARCH ---
        query = docs[0].vector
        results = engine.search(query, top_k=3)
        assert len(results) == 3
        assert results[0].uid == docs[0].uid  # nearest neighbour is itself

        # --- UPDATE (re-add with same UID) ---
        updated = VectorDocument(uid=docs[0].uid, vector=docs[1].vector, payload={"updated": True})
        engine.add([updated])
        assert engine._vector_store.ntotal == MIN_TRAIN  # count unchanged

        retrieved = engine.get_by_id(docs[0].uid)
        assert retrieved is not None
        assert retrieved.payload == {"updated": True}

        # --- DELETE ---
        engine.delete([docs[0].uid])
        assert engine._vector_store.ntotal == MIN_TRAIN - 1
        assert engine.get_by_id(docs[0].uid) is None


# -----------------------------------------------------------------------
# Persist and reload
# -----------------------------------------------------------------------


@pytest.mark.parametrize("index_type", IVF_TYPES)
def test_ivf_persist_and_reload(index_type: IndexType) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, index_type)
        docs = _make_docs(MIN_TRAIN)
        engine.add(docs)
        engine.persist()
        engine.close()

        # Reload from the same directory.
        engine2 = _make_engine(tmp, index_type)
        assert engine2._vector_store.ntotal == MIN_TRAIN
        result = engine2.get_by_id(docs[5].uid)
        assert result is not None
        assert result.uid == docs[5].uid


# -----------------------------------------------------------------------
# Rebuild integrity (forced via RuntimeError simulation)
# -----------------------------------------------------------------------


@pytest.mark.parametrize("index_type", IVF_TYPES)
def test_ivf_rebuild_preserves_data_integrity(index_type: IndexType) -> None:
    """Simulate a rebuild triggered by a failed remove_ids.

    We add MIN_TRAIN + 20 documents so that after deleting one the remaining
    count (MIN_TRAIN + 19) is still above the training threshold and the
    re-trained index remains usable.
    """
    n = MIN_TRAIN + 20
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, index_type)
        docs = _make_docs(n)
        engine.add(docs)

        # Patch remove() to always raise to force _rebuild_index path.
        original_remove = engine._vector_store.remove

        def _failing_remove(ids):
            raise RuntimeError("simulated failure")

        engine._vector_store.remove = _failing_remove
        try:
            engine.delete([docs[0].uid])
        finally:
            engine._vector_store.remove = original_remove

        assert engine._vector_store.ntotal == n - 1
        # All remaining docs should still be searchable.
        for doc in docs[1:4]:
            assert engine.get_by_id(doc.uid) is not None


# -----------------------------------------------------------------------
# IVF_PQ: exact vector reconstruction via SQLite BLOB
# -----------------------------------------------------------------------


def test_ivf_pq_get_by_id_returns_exact_vector() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = _make_engine(tmp, IndexType.IVF_PQ)
        docs = _make_docs(MIN_TRAIN)
        engine.add(docs)

        target = docs[10]
        retrieved = engine.get_by_id(target.uid)
        assert retrieved is not None
        # Exact vector preserved via SQLite BLOB (not lossy PQ reconstruction).
        assert retrieved.vector == pytest.approx(target.vector, abs=1e-6)


def test_ivf_pq_vector_blob_absent_for_non_pq_types() -> None:
    """FLAT index must not write vector BLOBs (vector column should be NULL)."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        engine = FaissEngine(tmp, DIM)  # default FLAT
        docs = _make_docs(5)
        engine.add(docs)

        db_path = str((engine._doc_store._db_path))  # type: ignore[attr-defined]
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT vector FROM documents").fetchall()
        conn.close()
        assert all(row[0] is None for row in rows), "FLAT index must not write vector BLOBs"


# -----------------------------------------------------------------------
# Backward-compatibility: default FLAT index unchanged
# -----------------------------------------------------------------------


def test_default_flat_index_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = FaissEngine(tmp, DIM)  # no IndexConfig → FLAT
        docs = _make_docs(5)
        engine.add(docs)
        assert engine._vector_store.ntotal == 5
        result = engine.search(docs[0].vector, top_k=1)
        assert result[0].uid == docs[0].uid
        engine.delete([docs[0].uid])
        assert engine._vector_store.ntotal == 4
