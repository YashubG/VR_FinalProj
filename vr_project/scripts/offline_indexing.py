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
    img:        Image.Image,
    rel_path:   str,
    item_id:    str,
    detector:   YOLODetector,
    clip_enc:   CLIPEncoder,
    captioner:  Optional[BLIP2Captioner],
    alpha:      float,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full offline pipeline for one image.

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
        caption  = captioner.generate_caption(cropped)
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
    """
    root = Path(root)

    # ── load models if not provided ───────────────────────────────────────────
    if detector is None:
        print("[Indexing] Loading YOLO detector ...")
        detector = YOLODetector()
    if clip_enc is None:
        print("[Indexing] Loading CLIP encoder ...")
        clip_enc = CLIPEncoder()
    if captioner is None and use_blip2 and alpha < 1.0:
        print("[Indexing] Loading BLIP-2 captioner ...")
        captioner = BLIP2Captioner()

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
    t0 = time.time()
    for i, (rel_path, item_id) in enumerate(tqdm(records, desc="Indexing")):
        if i in done_set:
            continue

        img_path = root / "Img/" rel_path
        if not img_path.exists():
            print(f"[Indexing] Warning: image not found: {img_path}")
            continue

        try:
            img = load_image(img_path)
            emb, meta = process_single_image(
                img, rel_path, item_id,
                detector, clip_enc, captioner, alpha
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
