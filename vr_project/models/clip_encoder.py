"""
models/clip_encoder.py
-----------------------
CLIP visual + text encoder, plus fine-tuning support.

Design decisions
----------------
* We use open_clip (https://github.com/mlfoundations/open_clip) rather than
  OpenAI's original clip package, because it supports:
  - More weight variants
  - Native torch training (no monkey-patching)
  - Checkpoint save/load out of the box
* Fine-tuning strategy (per spec):
  - Only the CLIP **vision encoder** is fine-tuned.
  - The text encoder is frozen (reduces GPU memory; text semantics are stable).
  - Only the last `train_last_n_blocks` transformer blocks are unfrozen;
    earlier blocks preserve general low-level features.
* Contrastive loss (InfoNCE / NT-Xent) with a learnable temperature is used.
  Positive pairs = two images of the same item_id.
* The fused embedding formula from the spec:
      v_i = α * φ_V(x̂_i) + (1−α) * φ_T(c_i),  ‖v_i‖ = 1
  is implemented in `fuse_embeddings()`.

Local save/load
---------------
  - `save_clip()` saves the full fine-tuned model state to a .pt file.
  - `load_clip()` restores from that file if it exists; otherwise downloads
    from open_clip hub and saves immediately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CLIP_LOCAL_PATH,
    CLIP_CHECKPOINT_DIR,
    EMBEDDING_DIM,
    ALPHA,
    TRAIN_LAST_N_BLOCKS,
    DEVICE,
)
from utils.image_utils import get_clip_transform


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_clip(model, save_path: Path) -> None:
    """Save CLIP state-dict to disk (model-architecture-independent format)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"[CLIPEncoder] Saved fine-tuned CLIP to {save_path}")


def load_clip_weights(model, load_path: Path) -> None:
    """Load a previously saved state-dict into `model` in-place."""
    state = torch.load(str(load_path), map_location="cpu")
    model.load_state_dict(state)
    print(f"[CLIPEncoder] Loaded CLIP weights from {load_path}")


def save_clip_checkpoint(model, epoch: int, checkpoint_dir: Path) -> None:
    """Save an epoch checkpoint (useful for resuming training)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"clip_epoch_{epoch:03d}.pt"
    torch.save(model.state_dict(), str(path))
    print(f"[CLIPEncoder] Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_last_n_vision_blocks(model, n: int) -> None:
    """
    Unfreeze only the last `n` transformer blocks of the CLIP vision encoder.

    Rationale: lower blocks learn low-level features (edges, textures) that
    are domain-agnostic — keeping them frozen prevents catastrophic forgetting
    and reduces memory.  Upper blocks encode high-level semantics that benefit
    from domain adaptation to fashion.
    """
    # open_clip vision transformer stores residual blocks in .visual.transformer.resblocks
    try:
        blocks = list(model.visual.transformer.resblocks)
    except AttributeError:
        # fallback for modified architectures
        blocks = []
        for name, module in model.visual.named_modules():
            if "resblocks" in name.lower() and "." not in name.split("resblocks")[-1][1:]:
                blocks.append(module)

    if not blocks:
        print("[CLIPEncoder] Warning: could not locate vision transformer blocks. "
              "Unfreezing entire visual encoder.")
        for p in model.visual.parameters():
            p.requires_grad_(True)
        return

    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)

    # Always unfreeze the final layer norm and projection
    for name in ["ln_post", "proj"]:
        module = getattr(model.visual, name, None)
        if module is not None:
            for p in (module.parameters() if isinstance(module, nn.Module)
                      else [module]):
                p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[CLIPEncoder] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f} %)")


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive loss (InfoNCE)
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for contrastive learning.

    Given a batch of (anchor, positive) image embedding pairs sharing an
    item_id, the loss pulls same-item embeddings together and pushes
    different-item embeddings apart.

    temperature: lower → sharper distribution → harder negatives emphasised.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb_a: torch.Tensor,   # (B, D) L2-normalised
        emb_b: torch.Tensor,   # (B, D) L2-normalised
    ) -> torch.Tensor:
        # Similarity matrix: (B, B)
        logits = emb_a @ emb_b.T / self.temperature
        labels = torch.arange(len(emb_a), device=emb_a.device)
        loss_a = F.cross_entropy(logits,   labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return (loss_a + loss_b) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CLIPEncoder class
# ─────────────────────────────────────────────────────────────────────────────

class CLIPEncoder:
    """
    Wrapper around open_clip providing:
      - image embedding
      - text embedding
      - fused embedding (α * img + (1-α) * text, L2-normalised)
      - fine-tuning setup (partial unfreeze of vision encoder)

    Parameters
    ----------
    local_finetuned_path : Path to a saved fine-tuned .pt state-dict.
                           If this file exists it is loaded after base init.
    model_name           : open_clip architecture string.
    pretrained           : open_clip weight set name.
    alpha                : Image/text fusion weight.
    train_last_n_blocks  : Number of vision transformer blocks to unfreeze.
    device               : 'cuda' or 'cpu'.
    save_after_load      : If True, save base weights locally on first download
                           (writes to local_finetuned_path's parent directory).
    """

    def __init__(
        self,
        local_finetuned_path: Path  = CLIP_LOCAL_PATH,
        model_name:           str   = CLIP_MODEL_NAME,
        pretrained:           str   = CLIP_PRETRAINED,
        alpha:                float = ALPHA,
        train_last_n_blocks:  int   = TRAIN_LAST_N_BLOCKS,
        device:               str   = DEVICE,
        save_after_load:      bool  = True,
    ) -> None:
        self.alpha   = alpha
        self.device  = device
        self._transform = get_clip_transform()
        self._model, self._tokenizer = self._load(
            local_finetuned_path, model_name, pretrained, device, save_after_load
        )
        self._train_last_n_blocks = train_last_n_blocks

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(
        self,
        local_finetuned_path: Path,
        model_name:           str,
        pretrained:           str,
        device:               str,
        save_after_load:      bool,
    ):
        import open_clip

        local_finetuned_path = Path(local_finetuned_path)
        base_path = local_finetuned_path.parent / "clip_base.pt"

        # Always load the base architecture + pretrained weights
        if base_path.exists():
            print(f"[CLIPEncoder] Loading base CLIP from {base_path}")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.load_state_dict(torch.load(str(base_path), map_location="cpu"))
        else:
            print(f"[CLIPEncoder] Downloading CLIP '{model_name}' ({pretrained}) ...")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            if save_after_load:
                base_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(base_path))
                print(f"[CLIPEncoder] Base CLIP saved to {base_path}")

        # If a fine-tuned checkpoint exists, overlay it
        if local_finetuned_path.exists():
            print(f"[CLIPEncoder] Applying fine-tuned weights from {local_finetuned_path}")
            state = torch.load(str(local_finetuned_path), map_location="cpu")
            model.load_state_dict(state)

        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, tokenizer

    # ── fine-tuning setup ─────────────────────────────────────────────────────

    def prepare_for_finetuning(self) -> None:
        """
        Set gradient flags for fine-tuning:
          - Freeze everything
          - Unfreeze last N vision blocks + projection
        """
        _freeze_all(self._model)
        _unfreeze_last_n_vision_blocks(self._model, self._train_last_n_blocks)

    def set_train_mode(self) -> None:
        self._model.train()

    def set_eval_mode(self) -> None:
        self._model.eval()

    def parameters(self):
        """Return only trainable parameters (used by the optimiser)."""
        return [p for p in self._model.parameters() if p.requires_grad]

    # ── embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        """
        Encode a single PIL image → L2-normalised numpy vector (EMBEDDING_DIM,).
        """
        x = self._transform(img).unsqueeze(0).to(self.device)
        feat = self._model.encode_image(x)
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_image_batch(
        self, imgs: List[Image.Image], batch_size: int = 64
    ) -> np.ndarray:
        """Encode a list of PIL images → (N, EMBEDDING_DIM) numpy array."""
        all_feats = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            tensors = torch.stack([self._transform(im) for im in batch]).to(self.device)
            feats   = self._model.encode_image(tensors)
            feats   = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
        return np.vstack(all_feats)

    def encode_image_train(self, imgs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch tensor during training (keeps gradients).
        imgs_tensor: (B, 3, H, W) already pre-processed.
        """
        feats = self._model.encode_image(imgs_tensor)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single caption → L2-normalised numpy vector."""
        tokens = self._tokenizer([text]).to(self.device)
        feat   = self._model.encode_text(tokens)
        feat   = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_text_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> np.ndarray:
        """Encode a list of captions → (N, EMBEDDING_DIM) numpy array."""
        all_feats = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i : i + batch_size]
            tokens = self._tokenizer(batch).to(self.device)
            feats  = self._model.encode_text(tokens)
            feats  = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
        return np.vstack(all_feats)

    # ── fusion ────────────────────────────────────────────────────────────────

    def fuse_embeddings(
        self,
        img_emb:  np.ndarray,
        text_emb: np.ndarray,
        alpha:    Optional[float] = None,
    ) -> np.ndarray:
        """
        Fuse image and text embeddings per Equation 1 of the spec:
            v_i = α * φ_V(x̂_i) + (1−α) * φ_T(c_i),  ‖v_i‖ = 1

        Parameters
        ----------
        img_emb  : (D,) or (N, D) image embedding.
        text_emb : (D,) or (N, D) text embedding.
        alpha    : Override the instance-level alpha if provided.
        """
        alpha = alpha if alpha is not None else self.alpha
        fused = alpha * img_emb + (1.0 - alpha) * text_emb
        # L2-normalise
        if fused.ndim == 1:
            norm = np.linalg.norm(fused)
            return fused / (norm + 1e-10)
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        return fused / (norms + 1e-10)

    def fuse_single(
        self,
        img: Image.Image,
        caption: str,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Convenience: encode + fuse a single image/caption pair."""
        img_emb  = self.encode_image(img)
        text_emb = self.encode_text(caption)
        return self.fuse_embeddings(img_emb, text_emb, alpha)
