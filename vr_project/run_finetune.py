#!/usr/bin/env python3
"""
run_finetune.py
---------------
CLI script: fine-tune CLIP on the training split.

Usage
-----
# Single seed
python run_finetune.py --seed 42 --epochs 10

# Multi-seed (for ablation reporting)
python run_finetune.py --multiseed --epochs 10

# Custom split file
python run_finetune.py --split-file data/splits/train.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASET_DIR, SPLIT_DIR, SEEDS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CLIP for product retrieval")
    p.add_argument("--root",       type=Path, default=DATASET_DIR)
    p.add_argument("--split-file", type=Path, default=None,
                   help="Path to train.txt (rel_path item_id per line)")
    p.add_argument("--split",      choices=["official", "csv", "scan"], default="official")
    p.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LEARNING_RATE)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--multiseed",  action="store_true",
                   help="Run training for all seeds in config.SEEDS")
    p.add_argument("--last-n-blocks", type=int, default=4,
                   help="Number of vision transformer blocks to unfreeze")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    # ── Load training records ─────────────────────────────────────────────────
    if args.split_file and args.split_file.exists():
        from data.dataset import load_split_csv
        train_records = load_split_csv(args.split_file)
    elif args.split == "official":
        from data.dataset import parse_official_splits
        splits = parse_official_splits(root)
        train_records = splits.get("train", [])
    elif args.split == "csv":
        from data.dataset import load_split_csv
        train_records = load_split_csv(SPLIT_DIR / "train.txt")
    else:
        from data.dataset import scan_directory
        train_records = scan_directory(root)

    if not train_records:
        print("ERROR: No training records found.")
        sys.exit(1)

    print(f"[run_finetune] {len(train_records):,} training images")

    # ── Train ─────────────────────────────────────────────────────────────────
    from scripts.finetune_clip import train_one_seed, run_multiseed_training

    if args.multiseed:
        results = run_multiseed_training(
            train_records,
            seeds       = SEEDS,
            root        = root,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            last_n_blocks = args.last_n_blocks,
        )
    else:
        results = train_one_seed(
            train_records,
            seed        = args.seed,
            root        = root,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            last_n_blocks = args.last_n_blocks,
        )

    print("\n[run_finetune] Training complete.")


if __name__ == "__main__":
    main()
