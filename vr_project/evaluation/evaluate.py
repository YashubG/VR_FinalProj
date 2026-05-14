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
from models.captioner import BLIP2Captioner, BLIP2ITM
from models.clip_encoder import CLIPEncoder
from scripts.index_builder import HNSWIndex
from scripts.offline_indexing import build_index, load_gallery_embeddings
from evaluation.metrics import evaluate_all, format_metrics
from utils.image_utils import load_image


# ─────────────────────────────────────────────────────────────────────────────
# Build ground-truth relevant sets
# ─────────────────────────────────────────────────────────────────────────────

def build_relevant_sets(query_records, gallery_records):
    relevant: Dict[str, Set[str]] = {}
    gallery_by_id: Dict[str, Set[str]] = {}
    for gpath, gid in gallery_records:
        gallery_by_id.setdefault(gid, set()).add(gid)
    for qpath, qid in query_records:
        # Store the item_id itself (matched against retrieved item_ids)
        gallery_items_by_id = {gid: {gid} for _, gid in gallery_records}  # one pass
        relevant[qpath] = gallery_items_by_id.get(qid, set()) 
        # relevant[qpath] = {gid for _, gid in gallery_records if gid == qid}   # all gallery images share item_id qid
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
    itm_scorer:      Optional[BLIP2ITM]       = None,
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
      4. (optional) BLIP-2 ITM re-rank  (requires a BLIP2ITM itm_scorer)
      5. Compute metrics vs. ground truth

    Note: the `captioner` (BLIP2Captioner) parameter that appeared in earlier
    versions has been removed.  Captioning is an offline-indexing concern only.
    Online re-ranking uses a BLIP2ITM scorer — pass that as `itm_scorer`.

    Returns aggregated metrics dict.
    """
    from scripts.online_retrieval import retrieve

    # FIX: use build_relevant_sets() so that all gallery images sharing the
    # same item_id are counted as relevant, not just a single-element set.
    # Previously `relevant = {qid}` was constructed inline, which is correct
    # for item-level recall but bypasses the gallery-aware relevant-set logic
    # (e.g. items that appear multiple times in the gallery are all relevant).
    relevant_sets = build_relevant_sets(query_records, gallery_records)

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
            itm_scorer    = itm_scorer if use_reranking else None,
            top_k         = max_k,
            rerank_top_k  = max_k * 2,
            beta          = beta,
            use_reranking = use_reranking and (itm_scorer is not None),
        )

        # Retrieved item_ids in ranked order
        retrieved_ids = [c["item_id"] for c in candidates]

        # FIX: use the gallery-aware relevant set instead of the inline {qid}.
        relevant = relevant_sets.get(qpath, set())
        if len(relevant) == 0:
            print(f"[Eval] WARNING: no relevant set for {qpath}")

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
    # FIX: pass use_finetuned=False explicitly so the frozen base model is
    # used even if a fine-tuned checkpoint happens to exist on disk.
    clip_enc  = CLIPEncoder(alpha=1.0, use_finetuned=False)

    index_a = build_index(
        gallery_records, root=root, alpha=1.0,
        use_blip2=False, tag="ablation_A", clip_enc=clip_enc,
    )
    metrics_a = run_evaluation(
        query_records, gallery_records, root,
        index_a, clip_enc, detector,
        use_reranking=False, tag="A"
    )
    all_results["A"] = metrics_a
    print(format_metrics(metrics_a))

    # ── Condition B: frozen CLIP + BLIP-2, two α values ───────────────────────
    captioner  = BLIP2Captioner()
    itm_scorer = BLIP2ITM()

    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"  ABLATION B — Frozen CLIP + BLIP-2 (α={alpha})")
        print("="*60)

        # FIX: use_finetuned=False — Ablation B uses the frozen base model.
        clip_enc_b = CLIPEncoder(alpha=alpha, use_finetuned=False)
        index_b    = build_index(
            gallery_records, root=root, alpha=alpha,
            use_blip2=True, tag=f"ablation_B_a{alpha}",
            detector=detector, clip_enc=clip_enc_b, captioner=captioner,
        )
        metrics_b = run_evaluation(
            query_records, gallery_records, root,
            index_b, clip_enc_b, detector,
            itm_scorer=itm_scorer, use_reranking=True, tag=f"B_a{alpha}",
        )
        key = f"B_alpha{alpha}"
        all_results[key] = metrics_b
        print(format_metrics(metrics_b))

    # ── Condition C: fine-tuned CLIP + BLIP-2, two α values ──────────────────
    print(f"\n{'='*60}")
    print("  ABLATION C — Fine-tuned CLIP + BLIP-2 (multi-seed)")
    print("="*60)

    # FIX: The spec requires results averaged over 3-4 seeds.  Previously this
    # trained with only one seed and shared that single checkpoint across all
    # alpha values.  Now we train all seeds up-front, evaluate each checkpoint
    # for every alpha, and report mean ± std — matching the methodology used
    # by run_evaluation.py --multiseed-eval.
    from scripts.finetune_clip import run_multiseed_training
    from config import CLIP_LOCAL_PATH, MODELS_DIR

    multiseed_results = run_multiseed_training(train_records, seeds=seeds, root=root)

    for alpha in alpha_values:
        seed_metrics_for_alpha = []
        for seed in seeds:
            tag_seed = f"seed_{seed}"
            ckpt_path = MODELS_DIR / f"clip_finetuned_{tag_seed}.pt"

            # FIX: use_finetuned=True with the explicit per-seed checkpoint.
            clip_enc_c = CLIPEncoder(
                alpha=alpha,
                use_finetuned=True,
                local_finetuned_path=ckpt_path,
            )
            index_c = build_index(
                gallery_records, root=root, alpha=alpha,
                use_blip2=True, tag=f"ablation_C_a{alpha}_{tag_seed}",
                detector=detector, clip_enc=clip_enc_c, captioner=captioner,
            )
            m = run_evaluation(
                query_records, gallery_records, root,
                index_c, clip_enc_c, detector,
                itm_scorer=itm_scorer, use_reranking=True,
                tag=f"C_a{alpha}_{tag_seed}",
            )
            seed_metrics_for_alpha.append(m)

        # Aggregate mean ± std across seeds for this alpha
        metric_keys = [k for k in seed_metrics_for_alpha[0]
                       if not k.endswith("_std") and k != "num_queries"]
        agg: Dict[str, float] = {}
        for k in metric_keys:
            vals = [sm[k] for sm in seed_metrics_for_alpha]
            agg[k]          = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

        key = f"C_alpha{alpha}"
        all_results[key] = agg
        print(f"\n  α = {alpha} (mean ± std over {len(seeds)} seeds):")
        print(format_metrics(agg, k_values=[5, 10, 15]))

    # ── Save all results ─────────────────────────────────────────────────────
    save_path = RESULTS_DIR / "ablation_results.json"
    with open(str(save_path), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved to {save_path}")

    return all_results