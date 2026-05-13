"""
data/dataset.py
---------------
DeepFashion In-Shop Clothes Retrieval dataset handling.

Dataset structure expected on disk
------------------------------------
data/deepfashion/
├── img/
│   └── <category>/
│       └── id_<item_id>/
│           └── *.jpg
└── Anno/
    ├── list_eval_partition.txt   (item_id, split: train/query/gallery)
    └── list_bbox_inshop.txt      (optional: ground-truth boxes)

OR a flat CSV/text layout:
data/splits/
├── train.txt
├── query.txt
└── gallery.txt

Each line: <relative_image_path> <item_id>

Design decisions
----------------
* A single `DeepFashionDataset` covers train, query, and gallery splits by
  setting the `split` flag.  This avoids code duplication.
* For training, pairs of same-item images are sampled using a
  `SameCategoryBatchSampler` (or random pair sampling) to feed the InfoNCE
  contrastive loss.
* The dataset can be initialised from:
    a) Official DeepFashion partition files (parse_official_splits())
    b) Pre-built CSV files (load_split_csv())
    c) A directory scan fallback (scan_directory()) for custom data
* Ground-truth lookup (item_id → [image_paths]) is exposed for evaluation.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from config import DATASET_DIR, SPLIT_DIR, IMAGE_SIZE
from utils.image_utils import get_clip_transform, load_image


# ─────────────────────────────────────────────────────────────────────────────
# Split parsing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_image_prefix(dataset_dir, sample_path):
    """
    The DeepFashion partition file stores paths like:
        img/WOMEN/Tees_Tanks/id_00002260/01_2_side.jpg
 
    But the actual files on disk may live under a different root depending
    on how the dataset was extracted.  Common layouts:
 
        dataset_dir / img / WOMEN / ...           <- official release
        dataset_dir / Img / img / WOMEN / ...     <- Kaggle zip extraction
        dataset_dir / images / img / WOMEN / ...  <- some mirrors
 
    We probe candidate prefixes with the first real path from the partition
    file and return the (prefix, strip) pair that resolves to an existing
    file, so no manual path editing is ever needed.
 
    Parameters
    ----------
    prefix : sub-directory to prepend (e.g. "Img")
    strip  : leading token to remove from the raw path before joining
             (e.g. "img/" avoids "Img/img/img/..." doubling)
    """
    candidates = [
        ("",        ""),        # official:  dataset_dir/img/...
        ("Img",     ""),        # Kaggle:    dataset_dir/Img/img/...
        ("Img/img", "img/"),    # alt Kaggle: strip leading "img/" then prepend Img/img
        ("images",  ""),
        ("img",     "img/"),
    ]
    for prefix, strip in candidates:
        test = sample_path
        if strip and test.startswith(strip):
            test = test[len(strip):]
        probe = (Path(dataset_dir) / prefix / test) if prefix else (Path(dataset_dir) / test)
        if probe.exists():
            print(f"[Dataset] Image root resolved: '{prefix or '<direct>'}' (strip='{strip}')")
            return prefix, strip
    print(f"[Dataset] WARNING: could not resolve '{sample_path}' under {dataset_dir}. "
          "Check your dataset layout.")
    return "", ""

def parse_official_splits(
    dataset_dir: Path = DATASET_DIR,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the official DeepFashion In-Shop partition file.
 
    Returns a dict:
        {"train": [(rel_path, item_id), ...],
         "query": [...],
         "gallery": [...]}
 
    The official format of list_eval_partition.txt is:
        <num_items>
        image_name item_id evaluation_status
        img/TOPS/id_00000001/01_1_front.jpg id_00000001 train
        ...
 
    Path resolution
    ---------------
    The partition file records paths relative to the dataset root, but the
    actual disk layout depends on how the dataset was extracted.
    We auto-detect the layout via _resolve_image_prefix() by probing with
    the first path in the file, then rewrite all stored paths consistently.
    This means the code works on Kaggle, the official release, and mirrors
    without any manual config changes.
    """
    partition_file = Path(dataset_dir) / "Anno" / "list_eval_partition.txt"
    splits: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
 
    if not partition_file.exists():
        print(f"[Dataset] Official partition file not found: {partition_file}")
        return splits
 
    with open(str(partition_file)) as f:
        lines = f.readlines()
 
    # Collect raw records (skip two header lines)
    raw = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            raw.append((parts[0], parts[1], parts[2].lower()))
 
    if not raw:
        return dict(splits)
 
    # Auto-detect prefix using the first path in the file
    prefix, strip = _resolve_image_prefix(dataset_dir, raw[0][0])
 
    for rel_path, item_id, split in raw:
        # Apply strip before prefix to avoid duplicate segments
        if strip and rel_path.startswith(strip):
            rel_path = rel_path[len(strip):]
        if prefix:
            rel_path = str(Path(prefix) / rel_path)
        splits[split].append((rel_path, item_id))
 
    for k, v in splits.items():
        print(f"[Dataset] '{k}': {len(v):,} images")
    return dict(splits)

def load_split_csv(split_file: Path) -> List[Tuple[str, str]]:
    """
    Load a simple two-column text file:
        <relative_image_path> <item_id>

    Created by `save_split_csv()` below or manually.
    """
    records = []
    with open(str(split_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                records.append((parts[0], parts[1]))
    return records


def save_split_csv(
    records: List[Tuple[str, str]],
    save_path: Path,
) -> None:
    """Write split records to a plain-text file for reproducibility."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(save_path), "w") as f:
        for rel_path, item_id in records:
            f.write(f"{rel_path} {item_id}\n")
    print(f"[Dataset] Split saved to {save_path} ({len(records):,} items)")


def scan_directory(
    root: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> List[Tuple[str, str]]:
    """
    Fallback: scan a directory tree where each sub-directory name is the item_id.

    Expected layout:
        root/<item_id>/<image_file>

    Returns [(relative_path_from_root, item_id), ...].
    """
    root = Path(root)
    records = []
    for item_dir in sorted(root.iterdir()):
        if not item_dir.is_dir():
            continue
        item_id = item_dir.name
        for img_file in sorted(item_dir.iterdir()):
            if img_file.suffix.lower() in extensions:
                rel_path = str(img_file.relative_to(root))
                records.append((rel_path, item_id))
    print(f"[Dataset] Scanned {len(records):,} images from {root}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth helper
# ─────────────────────────────────────────────────────────────────────────────

def build_item_to_paths(
    records: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Map item_id → list of relative image paths (used for GT evaluation)."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    for rel_path, item_id in records:
        mapping[item_id].append(rel_path)
    return dict(mapping)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DeepFashionDataset(Dataset):
    """
    PyTorch Dataset for DeepFashion In-Shop Clothes Retrieval.

    For training (mode='train'):
        __getitem__ returns (img_tensor, item_id_index) to be used by
        the pair sampler / contrastive loss.

    For indexing / evaluation (mode='gallery' or 'query'):
        __getitem__ returns (img_tensor, item_id_str, rel_path).

    Parameters
    ----------
    records     : List of (rel_path, item_id) tuples.
    root        : Root directory containing images.
    transform   : Torchvision transform applied to each image.
    mode        : 'train' | 'gallery' | 'query'
    """

    def __init__(
        self,
        records:   List[Tuple[str, str]],
        root:      Path = DATASET_DIR,
        transform = None,
        mode:      str  = "gallery",
    ) -> None:
        self.records   = records
        self.root      = Path(root)
        self.transform = transform or get_clip_transform()
        self.mode      = mode

        # Build item_id → integer index (for contrastive loss labels)
        unique_ids = sorted(set(r[1] for r in records))
        self._id_to_int: Dict[str, int] = {v: i for i, v in enumerate(unique_ids)}

        # item_id → list of dataset indices (for pair sampling)
        self.item_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, (_, item_id) in enumerate(records):
            self.item_to_indices[item_id].append(idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rel_path, item_id = self.records[idx]
        img_path = self.root / rel_path
        img = load_image(img_path)
        tensor = self.transform(img)

        if self.mode == "train":
            label = self._id_to_int[item_id]
            return tensor, label
        else:
            return tensor, item_id, rel_path

    def get_image(self, idx: int) -> Image.Image:
        """Return raw PIL image for BLIP-2 captioning (no transform)."""
        rel_path, _ = self.records[idx]
        return load_image(self.root / rel_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pair-based batch sampler for contrastive training
# ─────────────────────────────────────────────────────────────────────────────

class PairBatchSampler(Sampler):
    """
    Yields batches of indices where each sample is paired with another
    sample of the same item_id.

    This constructs batches of size 2*B:
        [anchor_0, ..., anchor_B, positive_0, ..., positive_B]
    where anchor_i and positive_i share the same item_id.

    The InfoNCE loss then treats index i and index i+B as a positive pair.

    Items with only one image are skipped to guarantee a valid positive.
    """

    def __init__(
        self,
        dataset:    DeepFashionDataset,
        batch_size: int = 32,
        drop_last:  bool = True,
    ) -> None:
        self.dataset    = dataset
        self.batch_size = batch_size   # number of pairs per batch (total = 2*B)
        self.drop_last  = drop_last

        # Filter to items with ≥ 2 images
        self.valid_items = [
            item_id for item_id, indices in dataset.item_to_indices.items()
            if len(indices) >= 2
        ]

    def __iter__(self):
        random.shuffle(self.valid_items)
        anchors, positives = [], []

        for item_id in self.valid_items:
            indices = self.dataset.item_to_indices[item_id]
            a, p = random.sample(indices, 2)
            anchors.append(a)
            positives.append(p)

            if len(anchors) == self.batch_size:
                yield anchors + positives
                anchors, positives = [], []

        if anchors and not self.drop_last:
            yield anchors + positives

    def __len__(self) -> int:
        n = len(self.valid_items)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_train_loader(
    records:    List[Tuple[str, str]],
    root:       Path = DATASET_DIR,
    batch_size: int  = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader using PairBatchSampler for contrastive training."""
    ds = DeepFashionDataset(records, root=root, mode="train")
    sampler = PairBatchSampler(ds, batch_size=batch_size)
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_gallery_loader(
    records:    List[Tuple[str, str]],
    root:       Path = DATASET_DIR,
    batch_size: int  = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader for gallery/query embedding extraction."""
    ds = DeepFashionDataset(records, root=root, mode="gallery")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
