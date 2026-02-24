import tempfile
import threading
import unittest

from faiss_storage_lib.core import VectorDocument
from faiss_storage_lib.engine import FaissEngine


class TestFaissEngineThreading(unittest.TestCase):
    def test_each_thread_gets_dedicated_sqlite_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=2)
            thread_conn_ids: list[int] = []
            lock = threading.Lock()

            def worker() -> None:
                engine._get_conn().execute("SELECT 1")
                conn_id = id(engine._get_conn())
                with lock:
                    thread_conn_ids.append(conn_id)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            main_conn_id = id(engine._get_conn())
            self.assertEqual(len(thread_conn_ids), 2)
            self.assertEqual(len(set(thread_conn_ids)), 2)
            self.assertNotIn(main_conn_id, thread_conn_ids)
            engine.close()

    def test_new_connection_enables_wal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=2)
            conn = engine._get_conn()
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]

            self.assertEqual(str(journal_mode).lower(), "wal")
            engine.close()

    def test_add_delete_and_persist_operations_use_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=2)

            original_next_int_id = engine._next_int_id

            def assert_locked_next_int_id() -> int:
                self.assertTrue(engine._write_lock.locked())
                return original_next_int_id()

            engine._next_int_id = assert_locked_next_int_id  # type: ignore[method-assign]
            engine.add([VectorDocument(uid="doc-1", vector=[0.1, 0.2], payload={"k": "v"})])

            original_fetch_int_ids = engine._fetch_int_ids

            def assert_locked_fetch_int_ids(uids: list[str]) -> dict[str, int]:
                self.assertTrue(engine._write_lock.locked())
                return original_fetch_int_ids(uids)

            engine._fetch_int_ids = assert_locked_fetch_int_ids  # type: ignore[method-assign]
            engine.delete(["doc-1"])

            original_write_index = engine._faiss.write_index

            def assert_locked_write_index(index, path):
                self.assertTrue(engine._write_lock.locked())
                return original_write_index(index, path)

            engine._faiss.write_index = assert_locked_write_index
            engine.persist()
            engine._faiss.write_index = original_write_index
            engine.close()

    def test_close_is_idempotent_for_current_thread_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=2)
            engine._get_conn().execute("SELECT 1")
            engine.close()
            engine.close()
            conn = engine._get_conn()
            self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)
            engine.close()


if __name__ == "__main__":
    unittest.main()
