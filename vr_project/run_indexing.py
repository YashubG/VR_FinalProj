#!/usr/bin/env python3
"""
run_indexing.py
---------------
CLI script: build the offline HNSW index from gallery images.

Usage
-----
# Build from official DeepFashion partition files
python run_indexing.py --root data/deepfashion --split official

# Build from pre-split CSV files
python run_indexing.py --root data/deepfashion --split csv

# Ablation A (vision only, α=1.0)
python run_indexing.py --alpha 1.0 --no-blip2 --tag ablation_A

# Ablation B (frozen + BLIP-2, α=0.6)
python run_indexing.py --alpha 0.6 --tag ablation_B

# Ablation C (fine-tuned + BLIP-2, α=0.6)
python run_indexing.py --alpha 0.6 --finetuned --tag ablation_C
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASET_DIR, SPLIT_DIR, ALPHA
from data.dataset import (
    parse_official_splits,
    load_split_csv,
    scan_directory,
    save_split_csv,
)
from scripts.offline_indexing import build_index


def parse_args():
    p = argparse.ArgumentParser(description="Build HNSW product index")
    p.add_argument("--root",    type=Path, default=DATASET_DIR,
                   help="Root directory containing images")
    p.add_argument("--split",   choices=["official", "csv", "scan"], default="official",
                   help="How to load splits")
    p.add_argument("--alpha",   type=float, default=ALPHA,
                   help="Image/text fusion weight (1.0 = vision only)")
    p.add_argument("--no-blip2", action="store_true",
                   help="Skip BLIP-2 captioning (ablation A)")
    p.add_argument("--finetuned", action="store_true",
                   help="Use fine-tuned CLIP checkpoint (ablation C)")
    p.add_argument("--tag",     type=str, default="gallery",
                   help="Namespace for saved index files")
    p.add_argument("--save-every", type=int, default=500,
                   help="Checkpoint frequency (images)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    # ── Load gallery records ──────────────────────────────────────────────────
    if args.split == "official":
        splits = parse_official_splits(root)
        gallery_records = splits.get("gallery", [])
        if not gallery_records:
            print("No gallery records found. Falling back to directory scan.")
            gallery_records = scan_directory(root)
    elif args.split == "csv":
        csv_path = SPLIT_DIR / "gallery.txt"
        gallery_records = load_split_csv(csv_path)
    else:
        gallery_records = scan_directory(root)

    if not gallery_records:
        print("ERROR: No gallery records found.")
        sys.exit(1)

    print(f"[run_indexing] {len(gallery_records):,} gallery images | "
          f"α={args.alpha} | blip2={not args.no_blip2} | "
          f"finetuned={args.finetuned}")

    # ── Optionally load fine-tuned CLIP ───────────────────────────────────────
    clip_enc = None
    if args.finetuned:
        from models.clip_encoder import CLIPEncoder
        from config import CLIP_LOCAL_PATH
        if not CLIP_LOCAL_PATH.exists():
            print(f"WARNING: Fine-tuned checkpoint not found at {CLIP_LOCAL_PATH}. "
                  "Run run_finetune.py first. Using base CLIP.")
        clip_enc = CLIPEncoder(alpha=args.alpha)

    # ── Build index ───────────────────────────────────────────────────────────
    index = build_index(
        records     = gallery_records,
        root        = root,
        alpha       = args.alpha,
        use_blip2   = not args.no_blip2,
        tag         = args.tag,
        save_every  = args.save_every,
        clip_enc    = clip_enc,
    )

    print(f"\n[run_indexing] Done. Index contains {len(index):,} vectors.")


if __name__ == "__main__":
    main()
