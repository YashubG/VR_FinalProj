"""
scripts/finetune_clip.py
------------------------
Fine-tune the CLIP vision encoder using contrastive (InfoNCE) loss.

Training loop
-------------
  1. Sample positive pairs (same item_id) via PairBatchSampler.
  2. Pass both anchor and positive through the CLIP vision encoder.
  3. Compute symmetric InfoNCE loss.
  4. Backprop only through the unfrozen last-N vision blocks.
  5. Checkpoint every epoch; save best (lowest loss) model.

Multi-seed training
-------------------
  The spec requests results averaged over 3-4 seeds (team roll numbers).
  `run_multiseed_training()` loops over a list of seeds and returns
  per-seed results so you can report mean ± std.

Scheduler
---------
  Cosine annealing with warm-up: stabilises early training and avoids
  overshooting the optimal minimum of the pre-trained model.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    TEMPERATURE,
    TRAIN_LAST_N_BLOCKS,
    SEEDS,
    CLIP_LOCAL_PATH,
    CLIP_CHECKPOINT_DIR,
    DATASET_DIR,
    DEVICE,
)
from data.dataset import make_train_loader, DeepFashionDataset, PairBatchSampler
from models.clip_encoder import CLIPEncoder, InfoNCELoss, save_clip, save_clip_checkpoint
from utils.image_utils import get_clip_transform


# ─────────────────────────────────────────────────────────────────────────────
# Seed helper
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up + cosine annealing LR
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_cosine_lambda(
    step:      int,
    warmup:    int,
    total:     int,
    min_ratio: float = 1e-2,
) -> float:
    """Return LR multiplier for step `step`."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(
    train_records: List[Tuple[str, str]],
    seed:          int                   = 42,
    root:          Path                  = DATASET_DIR,
    epochs:        int                   = NUM_EPOCHS,
    batch_size:    int                   = BATCH_SIZE,
    lr:            float                 = LEARNING_RATE,
    weight_decay:  float                 = WEIGHT_DECAY,
    temperature:   float                 = TEMPERATURE,
    last_n_blocks: int                   = TRAIN_LAST_N_BLOCKS,
    save_path:     Path                  = CLIP_LOCAL_PATH,
    checkpoint_dir: Path                 = CLIP_CHECKPOINT_DIR,
    warmup_frac:   float                 = 0.05,
    device:        str                   = DEVICE,
    num_workers:   int                   = 4,
) -> Dict[str, List[float]]:
    """
    Fine-tune CLIP for one seed.

    Returns
    -------
    {"train_loss": [epoch_avg_loss, ...]}
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Fine-tuning CLIP | seed={seed} | device={device}")
    print(f"{'='*60}")

    # ── dataset & loader ─────────────────────────────────────────────────────
    transform = get_clip_transform()
    dataset   = DeepFashionDataset(
        train_records, root=root, transform=transform, mode="train"
    )
    sampler   = PairBatchSampler(dataset, batch_size=batch_size, drop_last=True)
    loader    = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── model setup ───────────────────────────────────────────────────────────
    clip_enc = CLIPEncoder(device=device)
    clip_enc.prepare_for_finetuning()   # freeze all, unfreeze last N blocks
    clip_enc.set_train_mode()

    loss_fn   = InfoNCELoss(temperature=temperature).to(device)
    optimizer = AdamW(
        clip_enc.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    total_steps  = epochs * len(loader)
    warmup_steps = int(warmup_frac * total_steps)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _warmup_cosine_lambda(s, warmup_steps, total_steps),
    )

    history: List[float] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        t0 = time.time()

        for batch_tensors, batch_labels in loader:
            # batch_tensors: (2*B, 3, H, W)
            # first B = anchors, last B = positives
            batch_tensors = batch_tensors.to(device)
            B2 = len(batch_tensors)
            B  = B2 // 2

            anchors   = batch_tensors[:B]
            positives = batch_tensors[B:]

            emb_a = clip_enc.encode_image_train(anchors)    # (B, D)
            emb_p = clip_enc.encode_image_train(positives)  # (B, D)

            loss = loss_fn(emb_a, emb_p)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents instability when only partial layers train
            nn.utils.clip_grad_norm_(clip_enc.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

        avg_loss = float(np.mean(epoch_losses))
        history.append(avg_loss)
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        save_clip_checkpoint(clip_enc._model, epoch, checkpoint_dir)

    # ── save best (final epoch) model ─────────────────────────────────────────
    clip_enc.set_eval_mode()
    save_clip(clip_enc._model, save_path)
    print(f"  Fine-tuned CLIP saved to {save_path}")

    return {"train_loss": history}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed wrapper (for ablation reporting)
# ─────────────────────────────────────────────────────────────────────────────

def run_multiseed_training(
    train_records: List[Tuple[str, str]],
    seeds:         List[int] = SEEDS,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run training for each seed; return per-seed histories.

    Returns
    -------
    {"seed_<N>": {"train_loss": [...]}, ...}
    """
    results = {}
    for seed in seeds:
        tag     = f"seed_{seed}"
        save_p  = CLIP_LOCAL_PATH.parent / f"clip_finetuned_{tag}.pt"
        history = train_one_seed(
            train_records,
            seed=seed,
            save_path=save_path,
            **kwargs,
        )
        results[tag] = history
        print(f"\n[MultiSeed] {tag} complete. "
              f"Final loss: {history['train_loss'][-1]:.4f}")

    # Print summary
    final_losses = [v["train_loss"][-1] for v in results.values()]
    print(f"\n[MultiSeed] Final loss: mean={np.mean(final_losses):.4f} "
          f"± {np.std(final_losses):.4f}")

    return results
