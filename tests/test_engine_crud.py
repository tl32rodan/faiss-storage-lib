import tempfile
import unittest

from faiss_storage_lib.core import VectorDocument
from faiss_storage_lib.engine import FaissEngine


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


if __name__ == "__main__":
    unittest.main()
