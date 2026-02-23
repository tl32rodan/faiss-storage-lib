import tempfile
import unittest

from faiss_storage_lib.core.schema import VectorDocument
from faiss_storage_lib.engine.faiss_engine import FaissEngine


class TestFaissEngineGetById(unittest.TestCase):
    def test_get_by_id_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                doc = VectorDocument(
                    uid="doc_123",
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"meta": "test_data"},
                )
                engine.add([doc])

                result = engine.get_by_id("doc_123")

                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result.uid, "doc_123")
                self.assertEqual(result.payload, {"meta": "test_data"})
                self.assertEqual(len(result.vector), 4)
                self.assertAlmostEqual(result.vector[0], 0.1, places=6)
            finally:
                engine.close()

    def test_get_by_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                self.assertIsNone(engine.get_by_id("non_existent"))
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
