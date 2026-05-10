"""
scripts/index_builder.py
------------------------
HNSW-based Approximate Nearest Neighbour (ANN) index for fast retrieval.

Design decisions
----------------
* We use `hnswlib` — a pure-C++ HNSW implementation with a Python binding.
  It is significantly faster than FAISS for single-query lookup on CPU and
  requires no GPU for search (important for deployment).

* The index stores only embedding vectors.  All metadata (item_id, image
  path, caption) is stored separately in a Python dict pickled alongside
  the index file.  This keeps the index file small and avoids hnswlib's
  limited label API.

* Index parameters:
    M               = 32   (higher → better recall, more RAM/build time)
    ef_construction = 200  (higher → better index quality)
    ef_search       = 100  (higher → better recall at query time, slower)
  These are good defaults for ~100 k–1 M vectors; tune if needed.

* All vectors added to the index must be L2-normalised.  Because we use
  cosine similarity, and hnswlib's inner-product space on unit vectors
  equals cosine similarity, this gives correct cosine rankings without an
  extra normalisation step at query time.

Save / Load
-----------
  save_index() writes:
    <HNSW_INDEX_PATH>      — hnswlib binary index
    <HNSW_METADATA_PATH>   — pickled dict {int_label: {item_id, path, caption}}

  load_index() restores both files.  If they don't exist, returns None so
  the caller can rebuild.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    HNSW_INDEX_PATH,
    HNSW_METADATA_PATH,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    EMBEDDING_DIM,
    DEFAULT_TOP_K,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata type alias
# ─────────────────────────────────────────────────────────────────────────────
# label (int) → {"item_id": str, "path": str, "caption": str}
Metadata = Dict[int, Dict[str, str]]


# ─────────────────────────────────────────────────────────────────────────────
# HNSWIndex class
# ─────────────────────────────────────────────────────────────────────────────

class HNSWIndex:
    """
    Thin wrapper around hnswlib for product embedding search.

    Workflow
    --------
    1. Create empty index:       idx = HNSWIndex()
    2. Add vectors:              idx.add(embeddings, metadata_list)
    3. Save:                     idx.save()
    4. Later — load:             idx = HNSWIndex.load()
    5. Query:                    results = idx.search(query_vec, top_k=10)
    """

    def __init__(
        self,
        dim:              int   = EMBEDDING_DIM,
        M:                int   = HNSW_M,
        ef_construction:  int   = HNSW_EF_CONSTRUCTION,
        ef_search:        int   = HNSW_EF_SEARCH,
        index_path:       Path  = HNSW_INDEX_PATH,
        metadata_path:    Path  = HNSW_METADATA_PATH,
        max_elements:     int   = 1_000_000,
    ) -> None:
        self.dim             = dim
        self.M               = M
        self.ef_construction = ef_construction
        self.ef_search       = ef_search
        self.index_path      = Path(index_path)
        self.metadata_path   = Path(metadata_path)

        self._metadata: Metadata = {}
        self._index = self._create_index(max_elements)

    # ── internal creation ─────────────────────────────────────────────────────

    def _create_index(self, max_elements: int):
        import hnswlib
        index = hnswlib.Index(space="ip", dim=self.dim)   # "ip" = inner product
        index.init_index(
            max_elements=max_elements,
            M=self.M,
            ef_construction=self.ef_construction,
            random_seed=42,
        )
        index.set_ef(self.ef_search)
        return index

    # ── adding vectors ────────────────────────────────────────────────────────

    def add(
        self,
        embeddings:     np.ndarray,                     # (N, D) float32, L2-normalised
        metadata_list:  List[Dict[str, str]],           # length N
        start_label:    int = 0,
    ) -> None:
        """
        Add a batch of embeddings with associated metadata.

        Parameters
        ----------
        embeddings    : (N, D) float32 array; must already be L2-normalised.
        metadata_list : List of dicts, one per embedding.
                        Each dict should have keys: 'item_id', 'path', 'caption'.
        start_label   : Integer label assigned to the first embedding.
                        Use to append incrementally without collision.
        """
        N = len(embeddings)
        assert N == len(metadata_list), "embeddings and metadata must have same length"

        labels = np.arange(start_label, start_label + N, dtype=np.int64)
        self._index.add_items(embeddings.astype(np.float32), labels)

        for label, meta in zip(labels, metadata_list):
            self._metadata[int(label)] = meta

    # ── querying ──────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,     # (D,) or (1, D) float32, L2-normalised
        top_k:     int = DEFAULT_TOP_K,
    ) -> List[Dict]:
        """
        Search for the top_k nearest neighbours of a query embedding.

        Returns
        -------
        List of dicts (length top_k), each containing:
            {label, item_id, path, caption, score}
        Sorted by score descending (higher = more similar).
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        labels, distances = self._index.knn_query(
            query_vec.astype(np.float32), k=min(top_k, self._index.element_count)
        )

        # hnswlib inner-product distance = 1 - cosine_similarity for unit vectors.
        # Convert to similarity: sim = 1 - dist  (or directly use negative dist).
        results = []
        for label, dist in zip(labels[0], distances[0]):
            label = int(label)
            meta  = self._metadata.get(label, {})
            results.append({
                "label":   label,
                "item_id": meta.get("item_id", ""),
                "path":    meta.get("path",    ""),
                "caption": meta.get("caption", ""),
                "score":   float(1.0 - dist),   # cosine similarity
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_batch(
        self,
        query_vecs: np.ndarray,    # (Q, D)
        top_k:      int = DEFAULT_TOP_K,
    ) -> List[List[Dict]]:
        """Search for multiple queries at once."""
        return [self.search(q, top_k) for q in query_vecs]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        index_path:    Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Save the HNSW index binary and metadata pickle to disk."""
        index_path    = Path(index_path    or self.index_path)
        metadata_path = Path(metadata_path or self.metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self._index.save_index(str(index_path))
        with open(str(metadata_path), "wb") as f:
            pickle.dump(self._metadata, f)

        print(f"[HNSWIndex] Index saved to {index_path} "
              f"({self._index.element_count:,} vectors)")
        print(f"[HNSWIndex] Metadata saved to {metadata_path}")

    @classmethod
    def load(
        cls,
        index_path:    Path = HNSW_INDEX_PATH,
        metadata_path: Path = HNSW_METADATA_PATH,
        dim:           int  = EMBEDDING_DIM,
        ef_search:     int  = HNSW_EF_SEARCH,
        max_elements:  int  = 1_000_000,
    ) -> Optional["HNSWIndex"]:
        """
        Load a previously saved index from disk.

        Returns None if the files don't exist (caller should rebuild).
        """
        import hnswlib

        index_path    = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            print(f"[HNSWIndex] No saved index found at {index_path}. "
                  "Run the offline indexing pipeline to build it.")
            return None

        instance = cls.__new__(cls)
        instance.dim           = dim
        instance.ef_search     = ef_search
        instance.index_path    = index_path
        instance.metadata_path = metadata_path

        # Load hnswlib index
        index = hnswlib.Index(space="ip", dim=dim)
        index.load_index(str(index_path), max_elements=max_elements)
        index.set_ef(ef_search)
        instance._index = index

        with open(str(metadata_path), "rb") as f:
            instance._metadata = pickle.load(f)

        print(f"[HNSWIndex] Loaded index: {index.element_count:,} vectors from {index_path}")
        return instance

    # ── info ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._index.element_count

    def __repr__(self) -> str:
        return (f"HNSWIndex(dim={self.dim}, "
                f"vectors={len(self)}, "
                f"M={self.M}, "
                f"ef_construction={self.ef_construction}, "
                f"ef_search={self.ef_search})")
