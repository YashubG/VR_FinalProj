#!/usr/bin/env python3
"""
run_evaluation.py
-----------------
Demo Script for Batch Evaluation (Deliverable 3 per spec).

Given a folder of query images, this script:
  1. Loads the pre-built HNSW gallery index
  2. Runs the retrieval pipeline end-to-end on every query image
  3. Computes Recall@K, NDCG@K, mAP@K for K ∈ {5, 10, 15}
  4. Saves a CSV + JSON of results

Usage
-----
# Evaluate on official query split
python run_evaluation.py --root data/deepfashion --split official

# Evaluate on a folder of images (no GT labels → shows ranked results only)
python run_evaluation.py --query-dir /path/to/query_images --index-tag gallery

# Specify a particular alpha / ablation tag
python run_evaluation.py --split official --index-tag ablation_C_a0.6 --alpha 0.6

# Save results to a custom file
python run_evaluation.py --split official --out results/my_eval.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATASET_DIR,
    SPLIT_DIR,
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    TOP_K_VALUES,
    ALPHA,
    DEVICE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluation of retrieval pipeline")
    p.add_argument("--root",       type=Path, default=DATASET_DIR,
                   help="Dataset root (image files)")
    p.add_argument("--split",      choices=["official","csv","scan"], default="official",
                   help="Source of query/gallery split information")
    p.add_argument("--query-dir",  type=Path, default=None,
                   help="Directory of query images (overrides --split)")
    p.add_argument("--index-tag",  type=str, default="gallery",
                   help="Tag used when saving the index (must match run_indexing.py)")
    p.add_argument("--alpha",      type=float, default=ALPHA)
    p.add_argument("--no-rerank",  action="store_true",
                   help="Skip BLIP-2 ITM re-ranking for speed")
    p.add_argument("--top-k",      type=int,   nargs="+", default=TOP_K_VALUES)
    p.add_argument("--out",        type=Path,  default=None,
                   help="Output JSON path (default: results/<tag>_metrics.json)")
    p.add_argument("--finetuned",  action="store_true",
                   help="Load fine-tuned CLIP checkpoint")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Folder-of-images mode (no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_query_folder(
    query_dir: Path,
    index_tag: str,
    alpha:     float,
    top_k:     List[int],
    use_rerank: bool,
    finetuned:  bool,
) -> None:
    """
    Run retrieval on every image in a folder; print ranked results.
    No ground-truth metrics computed (no item_id labels available).
    """
    from models.detector import YOLODetector
    from models.clip_encoder import CLIPEncoder
    from models.captioner import BLIP2Captioner
    from scripts.index_builder import HNSWIndex
    from scripts.online_retrieval import retrieve
    from utils.image_utils import load_image

    # Load models
    detector = YOLODetector()
    clip_enc = CLIPEncoder(alpha=alpha)
    captioner = BLIP2Captioner() if use_rerank else None

    # Load index
    idx_path  = EMBEDDINGS_DIR / f"{index_tag}_hnsw.bin"
    meta_path = EMBEDDINGS_DIR / f"{index_tag}_metadata.pkl"
    index = HNSWIndex.load(idx_path, meta_path)
    if index is None:
        print(f"ERROR: Index not found at {idx_path}")
        sys.exit(1)

    # Run on each image
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    query_files = sorted(f for f in query_dir.iterdir() if f.suffix.lower() in exts)
    print(f"\nQuerying {len(query_files)} images ...")

    max_k = max(top_k)
    for qf in query_files:
        img = load_image(qf)
        results = retrieve(
            img, index, clip_enc, detector, captioner,
            top_k=max_k, use_reranking=use_rerank,
        )
        print(f"\n  Query: {qf.name}")
        for rank, r in enumerate(results[:max_k], 1):
            print(f"    [{rank:2d}] item={r['item_id']} "
                  f"score={r.get('final_score', r.get('score', 0)):.4f} "
                  f"path={r['path']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation with ground truth
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── No-GT mode: just a folder of query images ────────────────────────────
    if args.query_dir is not None:
        evaluate_query_folder(
            args.query_dir, args.index_tag, args.alpha,
            args.top_k, not args.no_rerank, args.finetuned,
        )
        return

    # ── Load query + gallery records ─────────────────────────────────────────
    root = Path(args.root)
    if args.split == "official":
        from data.dataset import parse_official_splits
        splits = parse_official_splits(root)
        query_records   = splits.get("query",   [])
        gallery_records = splits.get("gallery", [])
    elif args.split == "csv":
        from data.dataset import load_split_csv
        query_records   = load_split_csv(SPLIT_DIR / "query.txt")
        gallery_records = load_split_csv(SPLIT_DIR / "gallery.txt")
    else:
        from data.dataset import scan_directory
        all_records     = scan_directory(root)
        # No split info: use all as both query and gallery
        query_records   = all_records[:min(200, len(all_records))]
        gallery_records = all_records

    if not query_records:
        print("ERROR: No query records found.")
        sys.exit(1)

    print(f"[Eval] {len(query_records):,} queries | "
          f"{len(gallery_records):,} gallery images")

    # ── Load models ───────────────────────────────────────────────────────────
    from models.detector import YOLODetector
    from models.clip_encoder import CLIPEncoder
    from models.captioner import BLIP2Captioner
    from scripts.index_builder import HNSWIndex
    from evaluation.evaluate import run_evaluation

    detector  = YOLODetector()
    clip_enc  = CLIPEncoder(alpha=args.alpha)
    captioner = None if args.no_rerank else BLIP2Captioner()

    # ── Load index ────────────────────────────────────────────────────────────
    tag       = args.index_tag
    idx_path  = EMBEDDINGS_DIR / f"{tag}_hnsw.bin"
    meta_path = EMBEDDINGS_DIR / f"{tag}_metadata.pkl"
    index = HNSWIndex.load(idx_path, meta_path)

    if index is None:
        print(f"ERROR: Index '{tag}' not found. Run run_indexing.py first.")
        sys.exit(1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = run_evaluation(
        query_records   = query_records,
        gallery_records = gallery_records,
        root            = root,
        index           = index,
        clip_enc        = clip_enc,
        detector        = detector,
        captioner       = captioner,
        top_k_values    = args.top_k,
        use_reranking   = not args.no_rerank,
        tag             = tag,
    )

    # ── Print + save ─────────────────────────────────────────────────────────
    from evaluation.metrics import format_metrics
    print("\n" + "="*50)
    print(f"  Results — {tag}")
    print("="*50)
    print(format_metrics(metrics, args.top_k))

    out_path = args.out or RESULTS_DIR / f"{tag}_metrics.json"
    with open(str(out_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
