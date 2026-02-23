import importlib
import unittest

import faiss_storage_lib
from faiss_storage_lib import FaissEngine, IndexRegistry, VectorDocument, core, engine


class TestPublicApi(unittest.TestCase):
    def test_package_exports(self) -> None:
        self.assertIs(faiss_storage_lib.VectorDocument, VectorDocument)
        self.assertIs(faiss_storage_lib.FaissEngine, FaissEngine)
        self.assertIs(faiss_storage_lib.IndexRegistry, IndexRegistry)

    def test_module_exports(self) -> None:
        self.assertIn("VectorDocument", core.__all__)
        self.assertIn("FaissEngine", engine.__all__)
        self.assertIn("IndexRegistry", engine.__all__)

    def test_importable_by_package_name(self) -> None:
        imported = importlib.import_module("faiss_storage_lib")

        self.assertIs(imported, faiss_storage_lib)
        self.assertIs(imported.VectorDocument, VectorDocument)
        self.assertIs(imported.FaissEngine, FaissEngine)
        self.assertIs(imported.IndexRegistry, IndexRegistry)


if __name__ == "__main__":
    unittest.main()
