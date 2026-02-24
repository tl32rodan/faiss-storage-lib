import tempfile
import threading
import unittest

from faiss_storage_lib.engine import FaissEngine


class TestFaissEngineThreading(unittest.TestCase):
    def test_each_thread_gets_dedicated_sqlite_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=2)
            thread_conn_ids: list[int] = []
            lock = threading.Lock()

            def worker(uid: str) -> None:
                engine._get_conn().execute("SELECT 1")
                conn_id = id(engine._get_conn())
                with lock:
                    thread_conn_ids.append(conn_id)

            threads = [threading.Thread(target=worker, args=(f"t{idx}",)) for idx in range(2)]
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


if __name__ == "__main__":
    unittest.main()
