from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VectorDocument:
    """
    The atomic unit of storage.
    The 'Blind Vault' only accepts this format.
    """

    uid: str
    vector: List[float]
    payload: Dict[str, Any]
    score: Optional[float] = None
