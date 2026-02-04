# FAISS Storage Library (Blind Vault)

This repository provides a minimal FAISS-backed storage library focused solely on
storing vectors and payloads. The "Blind Vault" persona means the library only
understands **IDs** and **vectors**: you provide a vector to store, you provide an
ID to delete, and you search by vector similarity. No parsing, no embedding, no
application-specific logic.

## Core Concepts

- **VectorDocument** is the single atomic record stored and retrieved by the system.
- **FaissEngine** provides CRUD operations backed by a FAISS `IndexFlatL2` index
  wrapped in an `IndexIDMap`, plus a SQLite document store for persistence.
- **IndexRegistry** manages multiple named indices under a single storage root,
  enabling multi-tenant storage.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from src.core import VectorDocument
from src.engine import FaissEngine

engine = FaissEngine("./data/source_code", dimension=1536)
engine.add(
    [
        VectorDocument(
            uid="file::func",
            vector=[0.0] * 1536,
            payload={"meta": "example"},
        )
    ]
)
results = engine.search([0.0] * 1536, top_k=5)
doc = engine.get_by_id("file::func")
engine.persist()
engine.close()
```

## Project Structure

- `src/core/schema.py`: `VectorDocument` definition.
- `src/engine/faiss_engine.py`: FAISS + SQLite storage engine.
- `src/engine/registry.py`: multi-tenant index registry.

## Development

```bash
pip install -r requirements.txt
ruff check .
python -m unittest discover -s tests
```
