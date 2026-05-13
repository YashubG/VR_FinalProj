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
from models.captioner import BLIP2ITM

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
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Explicit path to a .pt CLIP checkpoint (e.g. "
                        "models/clip_finetuned_seed_42.pt). Overrides --finetuned.")
    p.add_argument("--multiseed-eval", action="store_true",
                   help="Evaluate all per-seed checkpoints and report mean±std.")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Rebuild HNSW index per seed during --multiseed-eval.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Folder-of-images mode (no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_query_folder(
    query_dir:       Path,
    index_tag:       str,
    alpha:           float,
    top_k:           List[int],
    use_rerank:      bool,
    checkpoint_path: Optional[Path] = None,
) -> None:
    """
    Run retrieval on every image in a folder; print ranked results.
    No ground-truth metrics computed (no item_id labels available).

    Parameters
    ----------
    checkpoint_path : Optional path to a fine-tuned CLIP .pt file.
                      If None the frozen base CLIP is used.
    """
    from models.detector import YOLODetector
    from models.clip_encoder import CLIPEncoder
    from models.captioner import BLIP2ITM
    from scripts.index_builder import HNSWIndex
    from scripts.online_retrieval import retrieve
    from utils.image_utils import load_image

    # Load models
    detector = YOLODetector()
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            print(f"ERROR: checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        clip_enc = CLIPEncoder(alpha=alpha, use_finetuned=True,
                               local_finetuned_path=checkpoint_path)
    else:
        clip_enc = CLIPEncoder(alpha=alpha, use_finetuned=False)

    itm_scorer = BLIP2ITM() if use_rerank else None

    # ── Load index ────────────────────────────────────────────────────────────
    idx_path  = EMBEDDINGS_DIR / f"{index_tag}_hnsw.bin"
    meta_path = EMBEDDINGS_DIR / f"{index_tag}_metadata.pkl"
    index = HNSWIndex.load(idx_path, meta_path)
    if index is None:
        print(f"ERROR: Index not found at {idx_path}")
        sys.exit(1)

    # ── Run on each image ─────────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    query_files = sorted(f for f in query_dir.iterdir() if f.suffix.lower() in exts)
    print(f"\nQuerying {len(query_files)} images ...")

    max_k = max(top_k)
    for qf in query_files:
        img = load_image(qf)
        results = retrieve(
            img, index, clip_enc, detector, itm_scorer=itm_scorer,
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
        # Resolve checkpoint: --checkpoint > --finetuned > base CLIP
        ckpt: Optional[Path] = None
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
        elif args.finetuned:
            from config import CLIP_LOCAL_PATH
            ckpt = CLIP_LOCAL_PATH
        evaluate_query_folder(
            args.query_dir, args.index_tag, args.alpha,
            args.top_k, not args.no_rerank,
            checkpoint_path=ckpt,
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
    from scripts.index_builder import HNSWIndex
    from evaluation.evaluate import run_evaluation

    detector = YOLODetector()

    # FIX: Only load BLIP2ITM when actually needed for the single-checkpoint path.
    # For --multiseed-eval, the itm_scorer is constructed inside _run_multiseed_eval
    # to keep all model-loading logic co-located there.
    # We still pass a (possibly None) itm_scorer for the single-checkpoint path.
    itm_scorer = None if args.no_rerank else BLIP2ITM()

    # ── Multi-seed evaluation mode ────────────────────────────────────────────
    # FIX: use args.multiseed_eval directly (argparse dest for --multiseed-eval)
    # instead of getattr() with a fallback, which masked typos silently.
    if args.multiseed_eval:
        _run_multiseed_eval(
            args, query_records, gallery_records, root, detector,
        )
        return

    # ── Single checkpoint ─────────────────────────────────────────────────────
    # --checkpoint overrides --finetuned: lets you point at any specific .pt file,
    # e.g. models/clip_finetuned_seed_42.pt produced by run_finetune.py --multiseed
    if args.checkpoint:
        clip_enc = CLIPEncoder(alpha=args.alpha, use_finetuned=True,
                               local_finetuned_path=args.checkpoint)
    elif args.finetuned:
        from config import CLIP_LOCAL_PATH
        clip_enc = CLIPEncoder(alpha=args.alpha, use_finetuned=True,
                            local_finetuned_path=CLIP_LOCAL_PATH)
    else:
        clip_enc = CLIPEncoder(alpha=args.alpha, use_finetuned=False)

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
        itm_scorer      = itm_scorer,
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


def _run_multiseed_eval(args, query_records, gallery_records, root, detector):
    """
    Evaluate every per-seed checkpoint saved by ``run_finetune.py --multiseed``
    and report mean ± std across seeds.

    Checkpoint naming convention (from finetune_clip.py):
        models/clip_finetuned_seed_<SEED>.pt

    If --rebuild-index is set, a fresh HNSW index is built from each seed's
    checkpoint (necessary for Ablation C where the gallery embeddings change
    with the model weights).  If not set, the existing shared index is reused
    (valid when CLIP is frozen and embeddings are identical across seeds).

    Changes vs. original
    --------------------
    * BLIP2ITM is instantiated once here (not in main()) — the multiseed path
      no longer receives itm_scorer from main() to keep concerns co-located.
    * BLIP2Captioner is instantiated once before the seed loop (not once per
      seed inside the loop), saving N-1 expensive model loads.
    * A shared CaptionCache is created once and passed to every build_index
      call so BLIP-2 captions generated on seed 1 are reused on seeds 2+.
    * num_queries is excluded from the aggregate dict so the notebook's
      ``{v:.4f}`` formatter doesn't print it as a spurious metric.
    """
    import glob
    import numpy as np
    from models.clip_encoder import CLIPEncoder
    from scripts.index_builder import HNSWIndex
    from scripts.offline_indexing import build_index, CaptionCache
    from evaluation.evaluate import run_evaluation
    from evaluation.metrics import format_metrics
    from config import MODELS_DIR

    # ── Discover per-seed checkpoints produced by run_finetune.py ────────────
    pattern = str(MODELS_DIR / "clip_finetuned_seed_*.pt")
    ckpt_files = sorted(glob.glob(pattern))
    if not ckpt_files:
        print(f"ERROR: No per-seed checkpoints found matching {pattern}")
        print("Run: python run_finetune.py --multiseed")
        sys.exit(1)

    print(f"[MultiSeedEval] Found {len(ckpt_files)} checkpoint(s): "
          f"{[Path(c).name for c in ckpt_files]}")

    use_reranking = not args.no_rerank

    # FIX: Load BLIP2ITM once here (not passed in from main()) so the
    # multiseed path is self-contained and main() doesn't pay the loading
    # cost for models it won't use.
    itm_scorer = BLIP2ITM() if use_reranking else None

    # FIX: Load BLIP2Captioner once before the loop (not once per seed).
    # Captioning depends only on the image, never on the CLIP checkpoint.
    captioner = None
    if args.rebuild_index and use_reranking:
        from models.captioner import BLIP2Captioner
        captioner = BLIP2Captioner()
        print("[MultiSeedEval] BLIP2Captioner loaded (shared across all seeds)")

    # FIX: Create a single shared CaptionCache so seed 1 pays the BLIP-2
    # cost and seeds 2+ get captions from disk at near-zero cost.
    caption_cache = CaptionCache() if (args.rebuild_index and use_reranking) else None
    if caption_cache is not None:
        print(f"[MultiSeedEval] {caption_cache.stats()}")

    all_metrics: dict = {}

    for ckpt_path in ckpt_files:
        ckpt_path = Path(ckpt_path)
        # Derive a readable tag, e.g. "seed_42" from "clip_finetuned_seed_42.pt"
        seed_tag = ckpt_path.stem.replace("clip_finetuned_", "")
        print(f"\n{'='*55}")
        print(f"  Evaluating checkpoint: {ckpt_path.name}  [{seed_tag}]")
        print("="*55)

        clip_enc = CLIPEncoder(
            alpha=args.alpha,
            use_finetuned=True,
            local_finetuned_path=ckpt_path,
        )

        # ── Build or load the gallery index for this seed ─────────────────────
        if args.rebuild_index:
            # Ablation C: embeddings change per seed → must rebuild index.
            idx_tag = f"{args.index_tag}_{seed_tag}"
            print(f"  Building index tag={idx_tag} ...")
            index = build_index(
                gallery_records,
                root=root,
                alpha=args.alpha,
                use_blip2=use_reranking,
                tag=idx_tag,
                detector=detector,
                clip_enc=clip_enc,
                # FIX: pass the shared captioner (not a new one per seed)
                captioner=captioner,
                # FIX: pass the shared cache so captions from seed 1 are
                # reused on all subsequent seeds
                caption_cache=caption_cache,
            )
        else:
            # Ablation B / frozen: all seeds share the same gallery index.
            idx_tag   = args.index_tag
            idx_path  = EMBEDDINGS_DIR / f"{idx_tag}_hnsw.bin"
            meta_path = EMBEDDINGS_DIR / f"{idx_tag}_metadata.pkl"
            index = HNSWIndex.load(idx_path, meta_path)
            if index is None:
                print(f"  ERROR: Index '{idx_tag}' not found.")
                print("  Hint: pass --rebuild-index to build one per seed,"
                      " or run run_indexing.py first.")
                sys.exit(1)

        # ── Run evaluation for this seed's checkpoint ─────────────────────────
        metrics = run_evaluation(
            query_records   = query_records,
            gallery_records = gallery_records,
            root            = root,
            index           = index,
            clip_enc        = clip_enc,
            detector        = detector,
            itm_scorer      = itm_scorer if use_reranking else None,
            top_k_values    = args.top_k,
            use_reranking   = use_reranking,
            tag             = seed_tag,
        )
        all_metrics[seed_tag] = metrics
        print(format_metrics(metrics, args.top_k))

    # ── Aggregate mean ± std across all seeds ─────────────────────────────────
    if not all_metrics:
        print("ERROR: No metrics collected. Check checkpoint paths.")
        sys.exit(1)

    print("\n" + "="*55)
    print(f"  AGGREGATE (mean ± std across {len(all_metrics)} seed(s))")
    print("="*55)

    first_metrics = next(iter(all_metrics.values()))
    # FIX: also exclude num_queries from the aggregate dict — the notebook
    # iterates aggregate with `{v:.4f}` which would print "200.0000" as if
    # it were a metric.
    metric_keys = [
        k for k in first_metrics
        if not k.endswith("_std") and k != "num_queries"
    ]

    agg: dict = {}
    for k in metric_keys:
        vals = [all_metrics[s][k] for s in all_metrics]
        agg[k]          = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
        print(f"  {k:<20} {agg[k]:.4f} ± {agg[f'{k}_std']:.4f}")

    out_path = args.out or RESULTS_DIR / f"{args.index_tag}_multiseed_metrics.json"
    with open(str(out_path), "w") as f:
        json.dump(
            {
                "seeds":    list(all_metrics.keys()),
                "per_seed": all_metrics,
                "aggregate": agg,
            },
            f,
            indent=2,
        )
    print(f"\n[MultiSeedEval] Saved to {out_path}")


if __name__ == "__main__":
    main()