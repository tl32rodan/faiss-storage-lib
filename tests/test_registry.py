import tempfile
import unittest

from faiss_storage_lib.engine import IndexRegistry


class TestIndexRegistry(unittest.TestCase):
    def test_registry_returns_cached_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = IndexRegistry(tmpdir)
            first = registry.get_index("source_code", dimension=4)
            second = registry.get_index("source_code", dimension=4)
            other = registry.get_index("issues", dimension=4)

            self.assertIs(first, second)
            self.assertIsNot(first, other)

            first.close()
            other.close()


if __name__ == "__main__":
    unittest.main()
