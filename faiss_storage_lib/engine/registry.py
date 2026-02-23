from __future__ import annotations

from pathlib import Path
from typing import Dict

from faiss_storage_lib.engine.faiss_engine import FaissEngine


class IndexRegistry:
    def __init__(self, base_storage_path: str) -> None:
        self.base_storage_path = Path(base_storage_path)
        self._indices: Dict[str, FaissEngine] = {}

    def get_index(self, name: str, dimension: int = 1536) -> FaissEngine:
        if name not in self._indices:
            index_path = self.base_storage_path / name
            self._indices[name] = FaissEngine(str(index_path), dimension)
        return self._indices[name]
