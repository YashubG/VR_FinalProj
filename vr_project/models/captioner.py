"""
models/captioner.py
-------------------
Caption generation and Image-Text Matching (ITM) using BLIP-2 throughout.

Two separate BLIP-2 model classes are used because the tasks require
different BLIP-2 architectures / checkpoints:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Task               │ Class           │ HF class                          │
  │                    │                 │ & checkpoint                      │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Caption generation │ BLIP2Captioner  │ Blip2ForConditionalGeneration     │
  │                    │                 │ Salesforce/blip2-opt-2.7b         │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ ITM re-ranking     │ BLIP2ITM        │ Blip2ForImageTextRetrieval        │
  │                    │                 │ Salesforce/blip2-itm-vit-g        │
  └──────────────────────────────────────────────────────────────────────────┘

Why two different BLIP-2 checkpoints?
--------------------------------------
Blip2ForConditionalGeneration couples the Q-Former to an autoregressive LLM
(OPT / FlanT5). It is trained to decode text token-by-token — it has no
classification head and cannot produce a matching score.

Blip2ForImageTextRetrieval couples the Q-Former to a lightweight ITM head
trained with a contrastive + matching objective. Its forward pass returns
`itm_score` — a (B, 2) no-match/match logit tensor — which is exactly what
we need for re-ranking. The LLM decoder is absent entirely, making it faster
and lighter than the generative variant.

Local saving
-------------
On first download from HuggingFace Hub, both models and processors are saved
to local directories derived from BLIP2_LOCAL_PATH. Subsequent runs load from
disk with no internet access required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image

from config import BLIP2_MODEL_NAME, BLIP2_LOCAL_PATH, DEVICE

# ─────────────────────────────────────────────────────────────────────────────
# Model identifiers
# ─────────────────────────────────────────────────────────────────────────────

# Generative model — used for caption generation
BLIP2_GEN_HUB_NAME   = BLIP2_MODEL_NAME                         # "Salesforce/blip2-opt-2.7b"
BLIP2_GEN_LOCAL_PATH = Path(BLIP2_LOCAL_PATH)                   # models/blip2/

# ITM model — Blip2ForImageTextRetrieval; lighter (no LLM decoder)
BLIP2_ITM_HUB_NAME   = "Salesforce/blip2-itm-vit-g"
BLIP2_ITM_LOCAL_PATH = Path(BLIP2_LOCAL_PATH).parent / "blip2_itm"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _local_or_hub(local_path: Path, hub_name: str) -> str:
    """Return the local directory if it contains a saved model, else the hub id."""
    local_path = Path(local_path)
    if (local_path / "config.json").exists():
        print(f"[Captioner] Loading from local path: {local_path}")
        return str(local_path)
    print(f"[Captioner] Local model not found. Will download '{hub_name}' ...")
    return hub_name


def _save_hf_model(model, processor, save_path: Path) -> None:
    """Persist a HuggingFace model + processor to disk for offline use."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))
    print(f"[Captioner] Saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Caption generator  (Blip2ForConditionalGeneration)
# ─────────────────────────────────────────────────────────────────────────────

class BLIP2Captioner:
    """
    Caption generation using Blip2ForConditionalGeneration.

    The Q-Former feeds visual tokens into an autoregressive LLM (OPT-2.7B by
    default) which decodes a natural-language product description.

    Captions produced here serve two roles in the pipeline:
      • Offline: fused with the CLIP image embedding
            v_i = α·φ_V(x̂_i) + (1−α)·φ_T(caption_i),  ‖v_i‖ = 1
      • Online: passed to BLIP2ITM for cross-modal re-ranking.

    Parameters
    ----------
    local_path      : Directory to look for / save the local checkpoint.
    hub_name        : HuggingFace hub id (fallback when local absent).
    device          : 'cuda' or 'cpu'.
    load_in_8bit    : Enable bitsandbytes 8-bit quantisation (GPU only).
                      Reduces VRAM from ~16 GB to ~4 GB with minimal quality loss.
    num_beams       : Beam search width. 1 = greedy (fast); 4 = higher quality.
    max_new_tokens  : Maximum caption length in tokens.
    save_after_load : Persist the downloaded model locally on first use.
    """

    def __init__(
        self,
        local_path:      Path = BLIP2_GEN_LOCAL_PATH,
        hub_name:        str  = BLIP2_GEN_HUB_NAME,
        device:          str  = DEVICE,
        load_in_8bit:    bool = False,
        num_beams:       int  = 1,
        max_new_tokens:  int  = 64,
        save_after_load: bool = True,
    ) -> None:
        self.device         = device
        self.num_beams      = num_beams
        self.max_new_tokens = max_new_tokens

        self._model, self._processor = self._load(
            Path(local_path), hub_name, device, load_in_8bit, save_after_load
        )

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self, local_path, hub_name, device, load_in_8bit, save_after_load):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        model_id    = _local_or_hub(local_path, hub_name)
        is_download = model_id == hub_name

        processor = Blip2Processor.from_pretrained(model_id)

        kwargs: dict = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        if load_in_8bit and device == "cuda":
            kwargs["load_in_8bit"] = True
            kwargs["device_map"]   = "auto"
        else:
            kwargs["device_map"] = None

        model = Blip2ForConditionalGeneration.from_pretrained(model_id, **kwargs)

        if not (load_in_8bit and device == "cuda"):
            model = model.to(device)

        model.eval()

        if is_download and save_after_load:
            _save_hf_model(model, processor, local_path)

        return model, processor

    # ── single caption ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_caption(self, img: Image.Image) -> str:
        """Generate a natural-language product description for one image."""
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        ids = self._model.generate(
            **inputs,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ── batch captions ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_captions_batch(
        self,
        images:     List[Image.Image],
        batch_size: int = 8,
    ) -> List[str]:
        """Generate captions for a list of images in mini-batches."""
        captions: List[str] = []
        for i in range(0, len(images), batch_size):
            batch  = images[i : i + batch_size]
            inputs = self._processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)
            ids = self._model.generate(
                **inputs,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
            captions.extend(
                c.strip()
                for c in self._processor.batch_decode(ids, skip_special_tokens=True)
            )
        return captions


# ─────────────────────────────────────────────────────────────────────────────
# 2. ITM scorer  (Blip2ForImageTextRetrieval)
# ─────────────────────────────────────────────────────────────────────────────

class BLIP2ITM:
    """
    Image-Text Matching using Blip2ForImageTextRetrieval.

    Architecture
    ------------
    This checkpoint couples the BLIP-2 Q-Former to a lightweight ITM
    classification head trained with contrastive + matching objectives.
    The LLM decoder is absent — the model is much smaller and faster than
    the generative variant and outputs a 2-class (no-match / match) logit
    tensor directly from the Q-Former output.

    forward() returns an object with:
        .itm_score : (B, 2) float tensor — no-match logit, match logit

    We take softmax over dim=-1 and return the match probability (column 1),
    giving a calibrated score in [0, 1].

    Parameters
    ----------
    local_path      : Directory to look for / save the local checkpoint.
    hub_name        : HuggingFace hub id.
    device          : 'cuda' or 'cpu'.
    save_after_load : Persist the downloaded model locally on first use.
    """

    def __init__(
        self,
        local_path:      Path = BLIP2_ITM_LOCAL_PATH,
        hub_name:        str  = BLIP2_ITM_HUB_NAME,
        device:          str  = DEVICE,
        save_after_load: bool = True,
    ) -> None:
        self.device = device
        self._model, self._processor = self._load(
            Path(local_path), hub_name, device, save_after_load
        )

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self, local_path, hub_name, device, save_after_load):
        from transformers import Blip2Processor, Blip2ForImageTextRetrieval

        model_id    = _local_or_hub(local_path, hub_name)
        is_download = model_id == hub_name

        processor = Blip2Processor.from_pretrained(model_id)
        model     = Blip2ForImageTextRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        if is_download and save_after_load:
            _save_hf_model(model, processor, local_path)

        return model, processor

    # ── single score ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def itm_score(self, query_img: Image.Image, caption: str) -> float:
        """
        Compute an ITM score between one image and one caption.

        Returns the softmax probability of the 'match' class, in [0, 1].
        Higher = more likely the caption describes the image.
        """
        inputs = self._processor(
            images=query_img,
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self._model(**inputs)
        # outputs.itm_score: (1, 2) — no-match logit, match logit
        return float(F.softmax(outputs.itm_score, dim=-1)[0, 1].item())

    # ── batched scoring ───────────────────────────────────────────────────────

    @torch.no_grad()
    def itm_scores_batch(
        self,
        query_img:  Image.Image,
        captions:   List[str],
        batch_size: int = 16,
    ) -> List[float]:
        """
        Score multiple captions against one query image in true mini-batches.

        Each mini-batch passes (image × B, captions[i:i+B]) through the
        Q-Former and ITM head together in a single forward pass, rather than
        calling the model once per caption.

        Parameters
        ----------
        query_img  : The query image (held constant across all captions).
        captions   : Candidate captions from retrieved gallery items.
        batch_size : Number of (image, caption) pairs per forward pass.

        Returns
        -------
        List of float scores aligned with `captions`, each in [0, 1].
        """
        all_scores: List[float] = []

        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i : i + batch_size]
            # Replicate the query image once per caption in this mini-batch
            batch_images   = [query_img] * len(batch_captions)

            inputs = self._processor(
                images=batch_images,
                text=batch_captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs = self._model(**inputs)
            # outputs.itm_score: (B, 2)
            probs = F.softmax(outputs.itm_score, dim=-1)   # (B, 2)
            all_scores.extend(probs[:, 1].tolist())         # match probabilities

        return all_scores