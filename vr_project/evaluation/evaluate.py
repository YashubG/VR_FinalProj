"""
evaluation/evaluate.py
-----------------------
End-to-end evaluation of the retrieval system.

This module:
  1. Encodes all query images → query embeddings
  2. Searches the HNSW gallery index for each query
  3. Computes Recall@K, NDCG@K, mAP@K for K ∈ {5,10,15}
  4. Prints and saves results to results/

Ablation conditions (from spec):
  A: Vision-only CLIP (α=1.0), frozen, no BLIP-2
  B: Frozen CLIP + frozen BLIP-2 (α ∈ [0,1])
  C: Fine-tuned CLIP + frozen BLIP-2 (α ∈ [0,1])

`run_ablation_study()` handles all three in sequence.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    DATASET_DIR,
    RESULTS_DIR,
    TOP_K_VALUES,
    DEFAULT_TOP_K,
    ALPHA,
    SEEDS,
    DEVICE,
)
from data.dataset import (
    parse_official_splits,
    load_split_csv,
    build_item_to_paths,
    DeepFashionDataset,
    SPLIT_DIR,
)
from models.detector import YOLODetector
from models.captioner import BLIP2Captioner
from models.clip_encoder import CLIPEncoder
from scripts.index_builder import HNSWIndex
from scripts.offline_indexing import build_index, load_gallery_embeddings
from evaluation.metrics import evaluate_all, format_metrics
from utils.image_utils import load_image


# ─────────────────────────────────────────────────────────────────────────────
# Build ground-truth relevant sets
# ─────────────────────────────────────────────────────────────────────────────

def build_relevant_sets(
    query_records:   List[Tuple[str, str]],
    gallery_records: List[Tuple[str, str]],
) -> Dict[str, Set[str]]:
    """
    For each query image, build the set of gallery item_ids that are relevant.

    Relevant = same item_id as the query, excluding the query image itself.

    Returns
    -------
    Dict mapping query_path → set of relevant item_ids in gallery.
    """
    # Map item_id → set of gallery paths (for GT lookup)
    gallery_item_to_paths: Dict[str, Set[str]] = {}
    for gpath, gid in gallery_records:
        gallery_item_to_paths.setdefault(gid, set()).add(gpath)

    relevant: Dict[str, Set[str]] = {}
    for qpath, qid in query_records:
        # All gallery items with same item_id are relevant
        gallery_paths = gallery_item_to_paths.get(qid, set())
        # We return item_ids as the relevant set (not paths) for metric computation
        relevant[qpath] = {qid} if gallery_paths else set()

    return relevant


# ─────────────────────────────────────────────────────────────────────────────
# Single evaluation run
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    query_records:   List[Tuple[str, str]],
    gallery_records: List[Tuple[str, str]],
    root:            Path,
    index:           HNSWIndex,
    clip_enc:        CLIPEncoder,
    detector:        YOLODetector,
    captioner:       Optional[BLIP2Captioner] = None,
    top_k_values:    List[int]                = TOP_K_VALUES,
    use_reranking:   bool                     = False,
    beta:            float                    = 0.5,
    tag:             str                      = "eval",
) -> Dict[str, float]:
    """
    Evaluate the retrieval system on query_records against gallery_records.

    Steps per query:
      1. YOLO crop
      2. CLIP encode
      3. HNSW top-K search
      4. (optional) BLIP-2 ITM re-rank
      5. Compute metrics vs. ground truth

    Returns aggregated metrics dict.
    """
    from scripts.online_retrieval import retrieve

    # Build ground-truth relevant set per query
    # relevant[qpath] = {item_id_that_matches}
    gallery_item_ids = {gpath: gid for gpath, gid in gallery_records}
    query_item_map   = {qpath: qid for qpath, qid in query_records}

    # item_id → list of gallery item_ids that are relevant
    # (In DeepFashion, relevant = same item_id)
    all_results = []

    for qpath, qid in tqdm(query_records, desc=f"Evaluating [{tag}]"):
        img_path = root / qpath
        if not img_path.exists():
            continue

        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"[Eval] Could not load {img_path}: {e}")
            continue

        max_k = max(top_k_values)
        candidates = retrieve(
            query_img     = img,
            index         = index,
            clip_enc      = clip_enc,
            detector      = detector,
            captioner     = captioner if use_reranking else None,
            top_k         = max_k,
            rerank_top_k  = max_k * 5,
            beta          = beta,
            use_reranking = use_reranking and (captioner is not None),
        )

        # Retrieved item_ids in ranked order
        retrieved_ids = [c["item_id"] for c in candidates]

        # Relevant = any gallery item with the same item_id
        relevant = {qid}   # item_id equality is the ground truth

        all_results.append({
            "query_item_id": qid,
            "query_path":    qpath,
            "retrieved":     retrieved_ids,
            "relevant":      relevant,
        })

    metrics = evaluate_all(all_results, k_values=top_k_values)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full ablation study
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    query_records:   List[Tuple[str, str]],
    gallery_records: List[Tuple[str, str]],
    train_records:   List[Tuple[str, str]],
    root:            Path = DATASET_DIR,
    alpha_values:    List[float] = [0.6, 0.8],  # two α values per spec
    seeds:           List[int]   = SEEDS[:2],
) -> Dict[str, Dict]:
    """
    Run all three ablation conditions (A, B, C) and report metrics.

    Returns
    -------
    {
      "A": {metrics},
      "B_alpha0.6": {metrics},
      "B_alpha0.8": {metrics},
      "C_alpha0.6": {metrics},
      "C_alpha0.8": {metrics},
    }
    """
    from scripts.finetune_clip import train_one_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Condition A: vision-only CLIP, frozen ─────────────────────────────────
    print("\n" + "="*60)
    print("  ABLATION A — Vision-only CLIP (α=1.0), frozen")
    print("="*60)

    detector  = YOLODetector()
    clip_enc  = CLIPEncoder(alpha=1.0)   # frozen (no fine-tuning called)

    index_a = build_index(
        gallery_records, root=root, alpha=1.0,
        use_blip2=False, tag="ablation_A"
    )
    metrics_a = run_evaluation(
        query_records, gallery_records, root,
        index_a, clip_enc, detector,
        use_reranking=False, tag="A"
    )
    all_results["A"] = metrics_a
    print(format_metrics(metrics_a))

    # ── Condition B: frozen CLIP + BLIP-2, two α values ───────────────────────
    captioner = BLIP2Captioner()

    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"  ABLATION B — Frozen CLIP + BLIP-2 (α={alpha})")
        print("="*60)

        clip_enc_b = CLIPEncoder(alpha=alpha)
        index_b    = build_index(
            gallery_records, root=root, alpha=alpha,
            use_blip2=True, tag=f"ablation_B_a{alpha}",
            detector=detector, clip_enc=clip_enc_b, captioner=captioner,
        )
        metrics_b = run_evaluation(
            query_records, gallery_records, root,
            index_b, clip_enc_b, detector, captioner,
            use_reranking=True, tag=f"B_a{alpha}",
        )
        key = f"B_alpha{alpha}"
        all_results[key] = metrics_b
        print(format_metrics(metrics_b))

    # ── Condition C: fine-tuned CLIP + BLIP-2, two α values ──────────────────
    print(f"\n{'='*60}")
    print("  ABLATION C — Fine-tuned CLIP + BLIP-2")
    print("="*60)

    # Fine-tune once (using first seed) and share across α values
    from config import CLIP_LOCAL_PATH
    train_one_seed(train_records, seed=seeds[0], save_path=CLIP_LOCAL_PATH)

    for alpha in alpha_values:
        clip_enc_c = CLIPEncoder(alpha=alpha)   # loads fine-tuned weights automatically
        index_c    = build_index(
            gallery_records, root=root, alpha=alpha,
            use_blip2=True, tag=f"ablation_C_a{alpha}",
            detector=detector, clip_enc=clip_enc_c, captioner=captioner,
        )
        metrics_c = run_evaluation(
            query_records, gallery_records, root,
            index_c, clip_enc_c, detector, captioner,
            use_reranking=True, tag=f"C_a{alpha}",
        )
        key = f"C_alpha{alpha}"
        all_results[key] = metrics_c
        print(f"\n  α = {alpha}:")
        print(format_metrics(metrics_c))

    # ── Save all results ─────────────────────────────────────────────────────
    save_path = RESULTS_DIR / "ablation_results.json"
    with open(str(save_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved to {save_path}")

    return all_results
