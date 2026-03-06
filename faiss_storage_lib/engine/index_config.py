from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IndexType(Enum):
    """Supported FAISS index strategies.

    Trade-offs summary:
        FLAT    – Exact L2 search; highest RAM, no training required.
        IVF_FLAT – Inverted-file index; faster search, same RAM as FLAT.
                   Requires training on ≥ 39 × nlist vectors.
        IVF_SQ  – IVF + scalar quantisation; ~4–8× RAM reduction.
                   Reconstruction is approximate (small quantisation error).
                   Requires training on ≥ 39 × nlist vectors.
        IVF_PQ  – IVF + product quantisation; ~8–32× RAM reduction.
                   Reconstruction is lossy; raw vectors are stored in SQLite
                   to preserve exact values for get_by_id() and rebuild().
                   Requires training on ≥ 39 × nlist vectors.
    """

    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_SQ = "ivf_sq"
    IVF_PQ = "ivf_pq"


@dataclass
class IndexConfig:
    """Configuration for the FAISS index created by FaissVectorStore.

    Attributes:
        index_type: Which index family to use (see IndexType).
        nlist:      Number of Voronoi cells (IVF only).
                    Rule of thumb: max(1, min(4096, 4 × √n)) where n is the
                    expected dataset size. Fixed at index creation time.
        nprobe:     Number of cells inspected per query (IVF only).
                    Higher → better recall, slower search. Tunable at runtime.
        sq_type:    ScalarQuantizer variant for IVF_SQ.
                    One of: "QT_8bit", "QT_4bit", "QT_fp16".
        pq_m:       Number of sub-quantizers for IVF_PQ.
                    The vector dimension must be divisible by pq_m.
        pq_nbits:   Bits per sub-quantizer code for IVF_PQ (default 8).
                    Each sub-quantizer has 2^pq_nbits centroids; training
                    requires at least 2^pq_nbits vectors (hard FAISS limit).
                    Use 4 for small datasets (16 centroids per sub-quantizer).
    """

    index_type: IndexType = IndexType.FLAT
    nlist: int = 100
    nprobe: int = 10
    sq_type: str = "QT_8bit"
    pq_m: int = 8
    pq_nbits: int = 8
