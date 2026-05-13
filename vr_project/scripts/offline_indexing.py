"""
scripts/offline_indexing.py
----------------------------
Offline indexing pipeline.

For each gallery image:
  1. YOLO  → crop to main product region
  2. BLIP-2 → generate a semantic caption
  3. CLIP   → fuse image + text embeddings  (v_i = α·φ_V + (1-α)·φ_T)
  4. HNSW   → add fused vector + metadata to index

The pipeline is checkpointed every `save_every` images so it can resume
after interruption without re-processing already-done images.

Ablation modes
--------------
  'A'  (α=1.0, no BLIP-2):  vision-only CLIP, no fine-tuning
  'B'  (α∈[0,1], frozen):   captioning gain without fine-tuning
  'C'  (α∈[0,1], fine-tuned CLIP):  full system

The mode is selected by passing `alpha` and `use_blip2` flags.

Caption caching
---------------
BLIP-2 captions depend only on the image content, not on the CLIP weights or
the fusion parameter α.  For Ablation C (multi-seed), the same gallery images
are indexed once per seed, yet the captions would be identical every time.
`CaptionCache` avoids this redundant work:

  * A single shared file  ``blip2_caption_cache.pkl``  (under EMBEDDINGS_DIR)
    maps  ``rel_path → caption``  and is read/written by every `build_index`
    call that uses BLIP-2.
  * On a cache hit the captioner is never called; on a miss the caption is
    generated, stored in the in-memory dict, and the file is flushed to disk
    immediately so progress survives interruptions.
  * The cache is keyed on the image's relative path (stable across seeds).
  * Pass ``caption_cache=None`` to `build_index` to disable caching
    (e.g. when deliberately regenerating captions after a dataset change).
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    DATASET_DIR,
    EMBEDDINGS_DIR,
    ALPHA,
    DEFAULT_TOP_K,
    DEVICE,
)
from models.detector import YOLODetector
from models.captioner import BLIP2Captioner
from models.clip_encoder import CLIPEncoder
from scripts.index_builder import HNSWIndex
from utils.image_utils import load_image


# ─────────────────────────────────────────────────────────────────────────────
# BLIP-2 caption cache
# ─────────────────────────────────────────────────────────────────────────────

# Default location for the shared caption cache file.
_DEFAULT_CAPTION_CACHE_PATH = EMBEDDINGS_DIR / "blip2_caption_cache.pkl"


class CaptionCache:
    """
    Persistent key-value store: ``rel_path (str) → caption (str)``.

    The backing file is a plain pickle dict so it is human-inspectable with
    standard Python tools (``pickle.load``).

    Design notes
    ------------
    * The cache is **shared across all ablation runs** — keys are image paths,
      which are independent of the CLIP checkpoint or alpha value.
    * ``flush_every`` controls how often dirty state is written to disk.
      The default (1) flushes after every new caption, which is safe against
      crashes at the cost of slightly more I/O.  Set higher (e.g. 50) if I/O
      is a bottleneck and your environment is stable.
    * Thread/process safety: not implemented.  The indexing pipeline is
      single-process, so this is fine.

    Parameters
    ----------
    cache_path  : Path to the pickle file.  Created on first write.
    flush_every : Number of new entries between automatic disk flushes.
                  Use 1 for maximum crash-safety, higher for performance.
    """

    def __init__(
        self,
        cache_path: Path = _DEFAULT_CAPTION_CACHE_PATH,
        flush_every: int = 1,
    ) -> None:
        self.cache_path  = Path(cache_path)
        self.flush_every = flush_every
        self._store: Dict[str, str] = {}
        self._dirty_since_flush = 0   # new entries since last flush
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing cache from disk, or start empty."""
        if self.cache_path.exists():
            with open(str(self.cache_path), "rb") as f:
                self._store = pickle.load(f)
            print(f"[CaptionCache] Loaded {len(self._store):,} cached captions "
                  f"from {self.cache_path}")
        else:
            self._store = {}
            print(f"[CaptionCache] No existing cache at {self.cache_path}. "
                  "Starting fresh.")

    def flush(self) -> None:
        """Write current state to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.cache_path), "wb") as f:
            pickle.dump(self._store, f)
        self._dirty_since_flush = 0

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, rel_path: str) -> Optional[str]:
        """Return the cached caption for ``rel_path``, or None on a miss."""
        return self._store.get(rel_path)

    def set(self, rel_path: str, caption: str) -> None:
        """
        Store a caption and flush to disk every ``flush_every`` new entries.
        """
        self._store[rel_path] = caption
        self._dirty_since_flush += 1
        if self._dirty_since_flush >= self.flush_every:
            self.flush()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, rel_path: str) -> bool:
        return rel_path in self._store

    def stats(self) -> str:
        return f"CaptionCache({len(self._store):,} entries, path={self.cache_path})"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (resume support)
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(tag: str) -> Path:
    return EMBEDDINGS_DIR / f"checkpoint_{tag}.pkl"


def _save_checkpoint(
    done_indices: List[int],
    embeddings:   List[np.ndarray],
    metadata:     List[Dict],
    tag:          str,
) -> None:
    data = {"done": done_indices, "embeddings": embeddings, "metadata": metadata}
    with open(str(_checkpoint_path(tag)), "wb") as f:
        pickle.dump(data, f)


def _load_checkpoint(tag: str) -> Optional[Dict]:
    cp = _checkpoint_path(tag)
    if cp.exists():
        with open(str(cp), "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_single_image(
    img:           Image.Image,
    rel_path:      str,
    item_id:       str,
    detector:      YOLODetector,
    clip_enc:      CLIPEncoder,
    captioner:     Optional[BLIP2Captioner],
    alpha:         float,
    caption_cache: Optional[CaptionCache] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full offline pipeline for one image.

    Caption caching
    ---------------
    If ``caption_cache`` is provided and already contains an entry for
    ``rel_path``, BLIP-2 is skipped and the stored caption is reused.
    This is the key optimisation for Ablation C: the same gallery images are
    indexed once per seed, but captions are generated only on the first seed
    and reused on all subsequent ones.

    Returns
    -------
    (embedding_vector, metadata_dict)
    embedding_vector : (D,) float32, L2-normalised
    metadata_dict    : {"item_id": str, "path": str, "caption": str}
    """
    # Step 1: YOLO crop
    cropped, _ = detector.crop_product(img)

    # Step 2 + 3: BLIP-2 caption + CLIP fusion
    if captioner is not None and alpha < 1.0:
        # ── caption: cache hit? ────────────────────────────────────────────
        if caption_cache is not None and rel_path in caption_cache:
            caption = caption_cache.get(rel_path)
        else:
            caption = captioner.generate_caption(cropped)
            if caption_cache is not None:
                caption_cache.set(rel_path, caption)   # store + auto-flush

        img_emb  = clip_enc.encode_image(cropped)
        text_emb = clip_enc.encode_text(caption)
        fused    = clip_enc.fuse_embeddings(img_emb, text_emb, alpha=alpha)
    else:
        # Vision-only (alpha == 1.0 or no captioner)
        caption = ""
        fused   = clip_enc.encode_image(cropped)

    meta = {"item_id": item_id, "path": rel_path, "caption": caption}
    return fused.astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────────────
# Main indexing function
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    records:       List[Tuple[str, str]],      # [(rel_path, item_id), ...]
    root:          Path                = DATASET_DIR,
    alpha:         float               = ALPHA,
    use_blip2:     bool                = True,
    index_save_path: Optional[Path]    = None,
    metadata_save_path: Optional[Path] = None,
    tag:           str                 = "gallery",
    save_every:    int                 = 500,
    detector:      Optional[YOLODetector]    = None,
    clip_enc:      Optional[CLIPEncoder]     = None,
    captioner:     Optional[BLIP2Captioner]  = None,
    caption_cache: Optional[CaptionCache]    = None,
    cache_path:    Optional[Path]            = None,
) -> HNSWIndex:
    """
    Build (or resume building) an HNSW index over a set of gallery images.

    Parameters
    ----------
    records            : List of (rel_path, item_id).
    root               : Root directory for image files.
    alpha              : Fusion weight (1.0 = vision only).
    use_blip2          : Whether to generate captions (False = ablation A).
    index_save_path    : Where to save the HNSW binary index.
    metadata_save_path : Where to save the metadata pickle.
    tag                : Checkpoint namespace (use different tags per ablation).
    save_every         : Checkpoint frequency (images).
    detector / clip_enc / captioner : Pass pre-loaded models to avoid
                         re-loading them inside this function.
    caption_cache      : A ``CaptionCache`` instance shared across indexing
                         runs.  When provided, BLIP-2 is only invoked for
                         images not already in the cache — all other images
                         reuse the stored caption string.

                         Pass ``None`` to disable caching entirely (e.g. to
                         force regeneration after a dataset update).

                         If ``use_blip2=True`` and ``caption_cache`` is not
                         explicitly supplied, a default ``CaptionCache`` is
                         created automatically and loaded from / saved to
                         ``cache_path`` (or ``EMBEDDINGS_DIR/blip2_caption_cache.pkl``
                         if ``cache_path`` is None).  This means caching is
                         **on by default** for any BLIP-2 indexing run — pass
                         ``caption_cache=None`` only to opt out.
    cache_path         : Override the default cache file location.  Ignored
                         when ``caption_cache`` is passed directly.
    """
    root = Path(root)

    # ── load models if not provided ───────────────────────────────────────────
    if detector is None:
        print("[Indexing] Loading YOLO detector ...")
        detector = YOLODetector()
    if clip_enc is None:
        print("[Indexing] Loading CLIP encoder (frozen base) ...")
        # NOTE: use_finetuned=False is explicit here so that the base model is
        # always used when no encoder is supplied. This prevents accidentally
        # loading a fine-tuned checkpoint that may exist at CLIP_LOCAL_PATH
        # for Ablation A / B indices.
        clip_enc = CLIPEncoder(use_finetuned=False)
    if captioner is None and use_blip2 and alpha < 1.0:
        print("[Indexing] Loading BLIP-2 captioner ...")
        captioner = BLIP2Captioner()

    # ── set up caption cache ──────────────────────────────────────────────────
    # Auto-create a CaptionCache when BLIP-2 is active and no cache object
    # was supplied.  Passing caption_cache=None explicitly disables caching.
    # The sentinel `_CACHE_NOT_SET` lets us distinguish "caller passed None"
    # (opt-out) from "caller didn't pass the argument" (use default).
    # We achieve this by using a keyword-argument sentinel below: if the
    # argument is missing from the call we default to the string sentinel
    # and auto-create; if it is explicitly None we skip creation.
    #
    # Implementation note: the function signature already has
    # ``caption_cache: Optional[CaptionCache] = None`` which means we cannot
    # distinguish between "not provided" and "explicitly None" at the Python
    # level without a sentinel.  We choose the pragmatic default: if
    # use_blip2 is True and caption_cache is None, auto-create one.  The
    # caller can always pass a pre-built CaptionCache instance to share it
    # across multiple build_index calls (recommended for multi-seed runs).
    if use_blip2 and alpha < 1.0 and caption_cache is None:
        resolved_cache_path = Path(cache_path) if cache_path else _DEFAULT_CAPTION_CACHE_PATH
        caption_cache = CaptionCache(cache_path=resolved_cache_path)

    if caption_cache is not None:
        print(f"[Indexing] {caption_cache.stats()}")

    # ── resume from checkpoint ────────────────────────────────────────────────
    checkpoint = _load_checkpoint(tag)
    if checkpoint:
        done_set   = set(checkpoint["done"])
        embeddings = checkpoint["embeddings"]
        metadata   = checkpoint["metadata"]
        print(f"[Indexing] Resuming from checkpoint: {len(done_set):,} images done")
    else:
        done_set   = set()
        embeddings = []
        metadata   = []

    # ── process images ────────────────────────────────────────────────────────
    cache_hits   = 0
    cache_misses = 0
    t0 = time.time()
    for i, (rel_path, item_id) in enumerate(tqdm(records, desc="Indexing")):
        if i in done_set:
            continue

        img_path = root / rel_path
        if not img_path.exists():
            print(f"[Indexing] Warning: image not found: {img_path}")
            continue

        # Track cache hit/miss for logging
        if caption_cache is not None and rel_path in caption_cache:
            cache_hits += 1
        elif caption_cache is not None:
            cache_misses += 1

        try:
            img = load_image(img_path)
            emb, meta = process_single_image(
                img, rel_path, item_id,
                detector, clip_enc, captioner, alpha,
                caption_cache=caption_cache,
            )
            embeddings.append(emb)
            metadata.append(meta)
            done_set.add(i)
        except Exception as e:
            print(f"[Indexing] Error on {rel_path}: {e}")
            continue

        if len(done_set) % save_every == 0:
            _save_checkpoint(list(done_set), embeddings, metadata, tag)

    elapsed = time.time() - t0
    print(f"[Indexing] Processed {len(embeddings):,} images in {elapsed:.1f}s")
    if caption_cache is not None:
        total_captioned = cache_hits + cache_misses
        if total_captioned > 0:
            hit_pct = 100.0 * cache_hits / total_captioned
            print(f"[Indexing] Caption cache: {cache_hits:,} hits / "
                  f"{cache_misses:,} misses  ({hit_pct:.1f}% hit rate)")
        # Final flush to ensure all new captions are persisted
        caption_cache.flush()
        print(f"[Indexing] Caption cache flushed → {caption_cache.cache_path}")

    # ── build HNSW index ──────────────────────────────────────────────────────
    emb_array = np.vstack(embeddings).astype(np.float32)
    index = HNSWIndex(max_elements=max(len(emb_array) + 100, 10_000))
    index.add(emb_array, metadata, start_label=0)

    # ── save index ────────────────────────────────────────────────────────────
    index_save_path    = index_save_path    or EMBEDDINGS_DIR / f"{tag}_hnsw.bin"
    metadata_save_path = metadata_save_path or EMBEDDINGS_DIR / f"{tag}_metadata.pkl"
    index.index_path    = Path(index_save_path)
    index.metadata_path = Path(metadata_save_path)
    index.save()

    # Clean up checkpoint
    cp = _checkpoint_path(tag)
    if cp.exists():
        cp.unlink()

    return index


# ─────────────────────────────────────────────────────────────────────────────
# Save gallery embeddings as numpy arrays (for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def save_gallery_embeddings(
    embeddings:  np.ndarray,
    metadata:    List[Dict],
    tag:         str = "gallery",
) -> None:
    """
    Save raw embedding arrays and metadata for later metric computation
    without rebuilding the index.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_DIR / f"{tag}_embeddings.npy"), embeddings)
    with open(str(EMBEDDINGS_DIR / f"{tag}_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"[Indexing] Saved {len(embeddings)} embeddings to {EMBEDDINGS_DIR}/{tag}_*")


def load_gallery_embeddings(tag: str = "gallery") -> Tuple[np.ndarray, List[Dict]]:
    """Load previously saved gallery embeddings and metadata."""
    emb  = np.load(str(EMBEDDINGS_DIR / f"{tag}_embeddings.npy"))
    with open(str(EMBEDDINGS_DIR / f"{tag}_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return emb, meta