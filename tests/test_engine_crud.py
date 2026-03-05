import sqlite3
import tempfile
import unittest

from faiss_storage_lib.core import VectorDocument
from faiss_storage_lib.engine import FaissEngine


class _LimitedVariableConnection:
    def __init__(self, conn, max_variables: int) -> None:
        self._conn = conn
        self._max_variables = max_variables

    def execute(self, sql, parameters=()):
        if len(parameters) > self._max_variables:
            raise sqlite3.OperationalError("too many SQL variables")
        return self._conn.execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        return self._conn.executemany(sql, seq_of_parameters)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._conn.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TestFaissEngineCrud(unittest.TestCase):
    def test_add_update_delete_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                doc = VectorDocument(
                    uid="A",
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"msg": "hello"},
                )
                engine.add([doc])
                results = engine.search([0.1, 0.2, 0.3, 0.4], top_k=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].uid, "A")
                self.assertEqual(results[0].payload["msg"], "hello")

                updated = VectorDocument(
                    uid="A",
                    vector=[0.9, 0.9, 0.9, 0.9],
                    payload={"msg": "updated"},
                )
                engine.add([updated])
                updated_results = engine.search([0.9, 0.9, 0.9, 0.9], top_k=1)
                self.assertEqual(len(updated_results), 1)
                self.assertEqual(updated_results[0].payload["msg"], "updated")

                engine.delete(["A"])
                deleted_results = engine.search([0.9, 0.9, 0.9, 0.9], top_k=1)
                self.assertEqual(deleted_results, [])
            finally:
                engine.close()

    def test_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                engine.add(
                    [
                        VectorDocument(
                            uid="persisted",
                            vector=[0.2, 0.1, 0.0, 0.3],
                            payload={"msg": "stored"},
                        )
                    ]
                )
                engine.persist()
            finally:
                engine.close()

            reloaded = FaissEngine(tmpdir, dimension=4)
            try:
                results = reloaded.search([0.2, 0.1, 0.0, 0.3], top_k=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].uid, "persisted")
                self.assertEqual(results[0].payload["msg"], "stored")
            finally:
                reloaded.close()

    def test_get_tracked_sources_and_batch_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                docs = [
                    VectorDocument(
                        uid="doc-1",
                        vector=[1.0, 0.0, 0.0, 0.0],
                        payload={"metadata": {"source": "alpha"}, "msg": "first"},
                    ),
                    VectorDocument(
                        uid="doc-2",
                        vector=[0.0, 1.0, 0.0, 0.0],
                        payload={"metadata": {"source": "alpha"}, "msg": "second"},
                    ),
                    VectorDocument(
                        uid="doc-3",
                        vector=[0.0, 0.0, 1.0, 0.0],
                        payload={"metadata": {"source": "beta"}, "msg": "third"},
                    ),
                ]
                engine.add(docs)

                tracked = engine.get_tracked_sources()
                self.assertEqual(set(tracked.keys()), {"alpha", "beta"})
                self.assertEqual(set(tracked["alpha"]), {"doc-1", "doc-2"})
                self.assertEqual(tracked["beta"], ["doc-3"])

                engine.delete(tracked["alpha"])

                self.assertIsNone(engine.get_by_id("doc-1"))
                self.assertIsNone(engine.get_by_id("doc-2"))
                self.assertIsNotNone(engine.get_by_id("doc-3"))

                post_delete_tracked = engine.get_tracked_sources()
                self.assertEqual(post_delete_tracked, {"beta": ["doc-3"]})

                deleted_search_results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=3)
                self.assertEqual([doc.uid for doc in deleted_search_results], ["doc-3"])
            finally:
                engine.close()

    def test_large_batch_add_and_delete_exceeds_sqlite_variable_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                conn = engine._doc_store._get_conn()
                engine._doc_store._thread_local.conn = _LimitedVariableConnection(conn, max_variables=100)

                docs = [
                    VectorDocument(
                        uid=f"doc-{idx}",
                        vector=[float(idx), 0.0, 0.0, 0.0],
                        payload={"metadata": {"source": "bulk"}, "msg": str(idx)},
                    )
                    for idx in range(300)
                ]

                engine.add(docs)
                tracked = engine.get_tracked_sources()
                self.assertEqual(len(tracked["bulk"]), 300)

                engine.delete([doc.uid for doc in docs])
                self.assertEqual(engine.get_tracked_sources(), {})
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
