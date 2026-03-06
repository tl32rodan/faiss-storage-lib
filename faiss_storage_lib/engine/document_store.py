from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from faiss_storage_lib.core.schema import VectorDocument

_SQL_VARIABLE_CHUNK_SIZE = 100


class SqliteDocumentStore:
    """Manages SQLite persistence: schema, upsert, delete, and all fetch queries."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._thread_local = threading.local()
        self.ensure_schema()

    # ------------------------------------------------------------------
    # IDocumentDatabase implementation
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    uid TEXT PRIMARY KEY,
                    int_id INTEGER UNIQUE,
                    payload TEXT NOT NULL,
                    vector BLOB
                )
                """
            )
            # Idempotent migration for databases created before the vector
            # BLOB column was introduced.
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN vector BLOB")
            except sqlite3.OperationalError:
                pass  # column already exists

    def upsert(self, rows: List[Tuple[str, int, str]]) -> None:
        conn = self._get_conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO documents (uid, int_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    int_id = excluded.int_id,
                    payload = excluded.payload
                """,
                rows,
            )

    def upsert_with_vectors(
        self,
        rows: List[Tuple[str, int, str, Optional[bytes]]],
    ) -> None:
        """Upsert rows that include a serialised vector BLOB (for IVF_PQ)."""
        conn = self._get_conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO documents (uid, int_id, payload, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    int_id = excluded.int_id,
                    payload = excluded.payload,
                    vector = excluded.vector
                """,
                rows,
            )

    def delete(self, uids: List[str]) -> None:
        conn = self._get_conn()
        with conn:
            for uid_chunk in self._iter_chunks(uids):
                placeholders = ", ".join("?" for _ in uid_chunk)
                conn.execute(
                    f"DELETE FROM documents WHERE uid IN ({placeholders})",
                    uid_chunk,
                )

    def fetch_by_uid(self, uid: str) -> Optional[sqlite3.Row]:
        cursor = self._get_conn().execute(
            "SELECT uid, int_id, payload FROM documents WHERE uid = ?",
            (uid,),
        )
        return cursor.fetchone()

    def fetch_by_int_ids(self, ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
        rows: Dict[int, sqlite3.Row] = {}
        conn = self._get_conn()
        for id_chunk in self._iter_chunks(ids):
            placeholders = ", ".join("?" for _ in id_chunk)
            cursor = conn.execute(
                f"SELECT uid, int_id, payload FROM documents WHERE int_id IN ({placeholders})",
                id_chunk,
            )
            rows.update({int(row["int_id"]): row for row in cursor.fetchall()})
        return rows

    def fetch_int_ids(self, uids: Sequence[str]) -> Dict[str, int]:
        if not uids:
            return {}
        rows: Dict[str, int] = {}
        conn = self._get_conn()
        for uid_chunk in self._iter_chunks(uids):
            placeholders = ", ".join("?" for _ in uid_chunk)
            cursor = conn.execute(
                f"SELECT uid, int_id FROM documents WHERE uid IN ({placeholders})",
                uid_chunk,
            )
            rows.update({row["uid"]: int(row["int_id"]) for row in cursor.fetchall()})
        return rows

    def next_int_id(self) -> int:
        cursor = self._get_conn().execute("SELECT COALESCE(MAX(int_id), -1) + 1 FROM documents")
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def fetch_all_uid_int_ids(self) -> List[sqlite3.Row]:
        cursor = self._get_conn().execute("SELECT uid, int_id FROM documents")
        return cursor.fetchall()

    def fetch_vector_blob(self, int_id: int) -> List[float]:
        """Return the exact float32 vector stored as a BLOB for the given int_id.

        Returns an empty list if no blob is stored (e.g. non-IVF_PQ index types).
        """
        cursor = self._get_conn().execute(
            "SELECT vector FROM documents WHERE int_id = ?",
            (int_id,),
        )
        row = cursor.fetchone()
        if row is None or row["vector"] is None:
            return []
        return np.frombuffer(row["vector"], dtype="float32").tolist()

    def get_tracked_sources(self) -> Dict[str, List[str]]:
        cursor = self._get_conn().execute("SELECT uid, payload FROM documents")
        grouped: DefaultDict[str, list[str]] = defaultdict(list)
        for row in cursor.fetchall():
            uid = row["uid"]
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            source = metadata.get("source") if isinstance(metadata, dict) else None
            if isinstance(source, str):
                grouped[source].append(uid)
        return dict(grouped)

    def close(self) -> None:
        if hasattr(self._thread_local, "conn"):
            self._thread_local.conn.close()
            del self._thread_local.conn

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._thread_local, "conn"):
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._thread_local.conn = conn
        return self._thread_local.conn

    def _iter_chunks(
        self, values: Sequence[str] | Sequence[int]
    ) -> Iterable[list[str] | list[int]]:
        for start in range(0, len(values), _SQL_VARIABLE_CHUNK_SIZE):
            yield list(values[start : start + _SQL_VARIABLE_CHUNK_SIZE])
