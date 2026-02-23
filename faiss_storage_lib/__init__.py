"""FAISS storage library for vector documents."""

from faiss_storage_lib.core import VectorDocument
from faiss_storage_lib.engine import FaissEngine, IndexRegistry

__all__ = ["FaissEngine", "IndexRegistry", "VectorDocument"]
