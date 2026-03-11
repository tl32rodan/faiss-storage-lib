import tempfile
import unittest

from faiss_storage_lib.core import VectorDocument
from faiss_storage_lib.engine import FaissEngine


class TestScoreNormalization(unittest.TestCase):
    def test_identical_vector_returns_score_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                engine.add(
                    [VectorDocument(uid="a", vector=[1.0, 0.0, 0.0, 0.0], payload={})]
                )
                results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=1)
                self.assertEqual(len(results), 1)
                self.assertAlmostEqual(results[0].score, 1.0, places=5)
            finally:
                engine.close()

    def test_distant_vector_returns_low_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                engine.add(
                    [VectorDocument(uid="a", vector=[1.0, 0.0, 0.0, 0.0], payload={})]
                )
                results = engine.search([0.0, 0.0, 0.0, 1.0], top_k=1)
                self.assertEqual(len(results), 1)
                # L2 distance = 2.0, score = 1/(1+2) = 0.333...
                self.assertAlmostEqual(results[0].score, 1.0 / 3.0, places=5)
            finally:
                engine.close()

    def test_scores_are_between_zero_and_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                docs = [
                    VectorDocument(uid="near", vector=[0.9, 0.1, 0.0, 0.0], payload={}),
                    VectorDocument(uid="far", vector=[0.0, 0.0, 0.0, 10.0], payload={}),
                ]
                engine.add(docs)
                results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=2)
                self.assertEqual(len(results), 2)
                for r in results:
                    self.assertGreater(r.score, 0.0)
                    self.assertLessEqual(r.score, 1.0)
            finally:
                engine.close()

    def test_closer_vector_has_higher_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = FaissEngine(tmpdir, dimension=4)
            try:
                docs = [
                    VectorDocument(uid="close", vector=[0.9, 0.1, 0.0, 0.0], payload={}),
                    VectorDocument(uid="far", vector=[0.0, 0.0, 1.0, 1.0], payload={}),
                ]
                engine.add(docs)
                results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=2)
                self.assertEqual(results[0].uid, "close")
                self.assertGreater(results[0].score, results[1].score)
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
