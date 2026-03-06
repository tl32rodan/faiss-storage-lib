from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from faiss_storage_lib.engine.faiss_engine import FaissEngine
from faiss_storage_lib.engine.index_config import IndexConfig


class IndexRegistry:
    def __init__(self, base_storage_path: str) -> None:
        self.base_storage_path = Path(base_storage_path)
        self._indices: Dict[str, FaissEngine] = {}

    def get_index(
        self,
        name: str,
        dimension: int = 1536,
        index_config: Optional[IndexConfig] = None,
    ) -> FaissEngine:
        if name not in self._indices:
            index_path = self.base_storage_path / name
            self._indices[name] = FaissEngine(str(index_path), dimension, index_config)
        return self._indices[name]
