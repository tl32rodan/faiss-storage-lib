from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence

import numpy as np

from faiss_storage_lib.core.schema import VectorDocument

if TYPE_CHECKING:
    import faiss


class FaissEngine:
    def __init__(self, index_dir: str, dimension: int) -> None:
        self.index_dir = Path(index_dir)
        self.dimension = dimension
        self.index_path = self.index_dir / "faiss.index"
        self.docstore_path = self.index_dir / "docstore.db"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        import faiss

        self._faiss = faiss
        self._index = self._load_or_create_index()
        self._conn = sqlite3.connect(self.docstore_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def add(self, documents: Iterable[VectorDocument]) -> None:
        doc_list = list(documents)
        if not doc_list:
            return
        existing_ids = self._fetch_int_ids([doc.uid for doc in doc_list])
        ids_to_remove = [existing_ids[doc.uid] for doc in doc_list if doc.uid in existing_ids]
        rebuild_needed = False
        if ids_to_remove:
            try:
                self._remove_ids(ids_to_remove)
            except RuntimeError:
                rebuild_needed = True
        next_id = self._next_int_id()
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
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO documents (uid, int_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    int_id = excluded.int_id,
                    payload = excluded.payload
                """,
                payload_rows,
            )
        if rebuild_needed:
            self._rebuild_index({doc.uid: doc for doc in doc_list})
        else:
            self._index.add_with_ids(self._prepare_vectors(vectors), self._prepare_ids(ids))

    def delete(self, uids: Iterable[str]) -> None:
        uid_list = list(uids)
        if not uid_list:
            return
        existing_ids = self._fetch_int_ids(uid_list)
        ids = [existing_ids[uid] for uid in uid_list if uid in existing_ids]
        rebuild_needed = False
        if ids:
            try:
                self._remove_ids(ids)
            except RuntimeError:
                rebuild_needed = True
        with self._conn:
            placeholders = ", ".join("?" for _ in uid_list)
            self._conn.execute(
                f"DELETE FROM documents WHERE uid IN ({placeholders})",
                uid_list,
            )
        if rebuild_needed:
            self._rebuild_index({})

    def search(self, query_vector: List[float], top_k: int) -> List[VectorDocument]:
        if top_k <= 0 or self._index.ntotal == 0:
            return []
        query = self._prepare_vectors([query_vector])
        distances, indices = self._index.search(query, top_k)
        ids = [int(idx) for idx in indices[0] if idx >= 0]
        if not ids:
            return []
        rows = self._fetch_rows_by_int_ids(ids)
        results: List[VectorDocument] = []
        for distance, int_id in zip(distances[0], indices[0]):
            if int_id < 0:
                continue
            row = rows.get(int(int_id))
            if row is None:
                continue
            payload = json.loads(row["payload"])
            vector = self._reconstruct_vector(int(int_id))
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
        """
        Retrieve a document by its unique ID, reconstructing the vector from the index.
        """
        cursor = self._conn.execute(
            "SELECT uid, int_id, payload FROM documents WHERE uid = ?",
            (uid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        int_id = int(row["int_id"])
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            payload = {}

        vector = []
        try:
            vector = self._reconstruct_vector(int_id)
        except RuntimeError:
            pass

        return VectorDocument(
            uid=row["uid"],
            vector=vector,
            payload=payload,
            score=None,
        )

    def persist(self) -> None:
        self._faiss.write_index(self._index, str(self.index_path))

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    uid TEXT PRIMARY KEY,
                    int_id INTEGER UNIQUE,
                    payload TEXT NOT NULL
                )
                """
            )

    def _load_or_create_index(self) -> "faiss.Index":
        if self.index_path.exists():
            return self._faiss.read_index(str(self.index_path))
        return self._create_index()

    def _create_index(self) -> "faiss.Index":
        base_index = self._faiss.IndexFlatL2(self.dimension)
        return self._faiss.IndexIDMap(base_index)

    def _prepare_vectors(self, values: List[List[float]]) -> np.ndarray:
        return np.array(values, dtype="float32")

    def _prepare_ids(self, values: List[int]) -> np.ndarray:
        return np.array(values, dtype="int64")

    def _remove_ids(self, ids: List[int]) -> None:
        selector = self._faiss.IDSelectorBatch(self._prepare_ids(ids))
        self._index.remove_ids(selector)

    def _next_int_id(self) -> int:
        cursor = self._conn.execute("SELECT COALESCE(MAX(int_id), -1) + 1 FROM documents")
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def _fetch_int_ids(self, uids: Sequence[str]) -> Dict[str, int]:
        if not uids:
            return {}
        placeholders = ", ".join("?" for _ in uids)
        cursor = self._conn.execute(
            f"SELECT uid, int_id FROM documents WHERE uid IN ({placeholders})",
            list(uids),
        )
        return {row["uid"]: int(row["int_id"]) for row in cursor.fetchall()}

    def _fetch_rows_by_int_ids(self, ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
        placeholders = ", ".join("?" for _ in ids)
        cursor = self._conn.execute(
            f"SELECT uid, int_id, payload FROM documents WHERE int_id IN ({placeholders})",
            list(ids),
        )
        return {int(row["int_id"]): row for row in cursor.fetchall()}

    def _reconstruct_vector(self, int_id: int) -> List[float]:
        return self._reconstruct_vector_from(self._index, int_id)

    def _reconstruct_vector_from(self, index: "faiss.Index", int_id: int) -> List[float]:
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

    def _rebuild_index(self, overrides: Dict[str, VectorDocument]) -> None:
        old_index = self._index
        self._index = self._create_index()
        cursor = self._conn.execute("SELECT uid, int_id FROM documents")
        vectors: List[List[float]] = []
        ids: List[int] = []
        for row in cursor.fetchall():
            uid = row["uid"]
            int_id = int(row["int_id"])
            if uid in overrides:
                vector = overrides[uid].vector
            else:
                vector = self._reconstruct_vector_from(old_index, int_id)
                if not vector:
                    continue
            vectors.append(vector)
            ids.append(int_id)
        if vectors:
            self._index.add_with_ids(self._prepare_vectors(vectors), self._prepare_ids(ids))
