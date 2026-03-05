from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, List, Optional

from faiss_storage_lib.core.schema import VectorDocument
from faiss_storage_lib.engine.document_store import SqliteDocumentStore
from faiss_storage_lib.engine.interfaces import IDocumentDatabase, IVectorIndex
from faiss_storage_lib.engine.vector_store import FaissVectorStore


class FaissEngine:
    """
    StorageFacade: coordinates VectorStore and DocumentStore under a single write lock.

    Accepts optional IVectorIndex and IDocumentDatabase via dependency injection for
    testability and future extensibility (e.g. swapping FAISS for pgvector, SQLite for
    PostgreSQL).
    """

    def __init__(
        self,
        index_dir: str,
        dimension: int,
        *,
        vector_store: Optional[IVectorIndex] = None,
        doc_store: Optional[IDocumentDatabase] = None,
    ) -> None:
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        self._index_path = str(index_path / "faiss.index")
        self._vector_store: IVectorIndex = vector_store or FaissVectorStore(index_path, dimension)
        self._doc_store: IDocumentDatabase = doc_store or SqliteDocumentStore(
            index_path / "docstore.db"
        )
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, documents: Iterable[VectorDocument]) -> None:
        with self._write_lock:
            doc_list = list(documents)
            if not doc_list:
                return
            existing_ids = self._doc_store.fetch_int_ids([doc.uid for doc in doc_list])
            ids_to_remove = [existing_ids[doc.uid] for doc in doc_list if doc.uid in existing_ids]
            rebuild_needed = False
            if ids_to_remove:
                try:
                    self._vector_store.remove(ids_to_remove)
                except RuntimeError:
                    rebuild_needed = True
            next_id = self._doc_store.next_int_id()
            vectors: List[List[float]] = []
            ids: List[int] = []
            payload_rows = []
            for doc in doc_list:
                int_id = existing_ids.get(doc.uid)
                if int_id is None:
                    int_id = next_id
                    next_id += 1
                ids.append(int_id)
                vectors.append(doc.vector)
                payload_rows.append((doc.uid, int_id, json.dumps(doc.payload)))
            self._doc_store.upsert(payload_rows)
            if rebuild_needed:
                uid_int_ids = {row[0]: row[1] for row in payload_rows}
                all_rows = self._doc_store.fetch_all_uid_int_ids()
                uid_int_ids = {row["uid"]: int(row["int_id"]) for row in all_rows}
                self._vector_store.rebuild({doc.uid: doc for doc in doc_list}, uid_int_ids)
            else:
                import numpy as np

                self._vector_store.add(
                    np.array(vectors, dtype="float32"),
                    np.array(ids, dtype="int64"),
                )

    def delete(self, uids: list[str]) -> None:
        with self._write_lock:
            if not uids:
                return
            existing_ids = self._doc_store.fetch_int_ids(uids)
            ids = [existing_ids[uid] for uid in uids if uid in existing_ids]
            rebuild_needed = False
            if ids:
                try:
                    self._vector_store.remove(ids)
                except RuntimeError:
                    rebuild_needed = True
            self._doc_store.delete(uids)
            if rebuild_needed:
                all_rows = self._doc_store.fetch_all_uid_int_ids()
                uid_int_ids = {row["uid"]: int(row["int_id"]) for row in all_rows}
                self._vector_store.rebuild({}, uid_int_ids)

    def search(self, query_vector: List[float], top_k: int) -> List[VectorDocument]:
        if top_k <= 0 or self._vector_store.ntotal == 0:
            return []
        import numpy as np

        query = np.array([query_vector], dtype="float32")
        distances, indices = self._vector_store.search(query, top_k)
        ids = [int(idx) for idx in indices[0] if idx >= 0]
        if not ids:
            return []
        rows = self._doc_store.fetch_by_int_ids(ids)
        results: List[VectorDocument] = []
        for distance, int_id in zip(distances[0], indices[0]):
            if int_id < 0:
                continue
            row = rows.get(int(int_id))
            if row is None:
                continue
            payload = json.loads(row["payload"])
            vector = self._vector_store.reconstruct(int(int_id))
            results.append(
                VectorDocument(
                    uid=row["uid"],
                    vector=vector,
                    payload=payload,
                    score=float(distance),
                )
            )
        return results

    def get_by_id(self, uid: str) -> VectorDocument | None:
        row = self._doc_store.fetch_by_uid(uid)
        if row is None:
            return None
        int_id = int(row["int_id"])
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}
        vector: List[float] = []
        try:
            vector = self._vector_store.reconstruct(int_id)
        except RuntimeError:
            pass
        return VectorDocument(uid=row["uid"], vector=vector, payload=payload, score=None)

    def get_tracked_sources(self) -> dict[str, list[str]]:
        return self._doc_store.get_tracked_sources()

    def persist(self) -> None:
        with self._write_lock:
            self._vector_store.persist(self._index_path)

    def close(self) -> None:
        self._doc_store.close()
