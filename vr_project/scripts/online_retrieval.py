"""
scripts/online_retrieval.py
----------------------------
Online retrieval pipeline.

Given a user query image:
  Step 1. YOLO         → crop main product region
  Step 2. CLIP visual  → encode cropped image
  Step 3. HNSW ANN     → retrieve top-K candidates by cosine similarity
  Step 4. BLIP-2 ITM   → re-rank candidates using image–text matching scores

The pipeline is stateless (no global mutable state); all model objects are
passed as arguments so the Streamlit app can load them once and reuse.

Re-ranking rationale
--------------------
Cosine similarity between CLIP embeddings captures visual + semantic
similarity but can be noisy for fine-grained fashion retrieval.  BLIP-2
ITM scores how well a candidate's caption matches the query image, adding
a cross-modal reasoning signal that often corrects near-miss errors from
ANN retrieval.

We combine both signals:
    final_score = β * cosine_score + (1-β) * itm_score
where β = 0.5 by default (equal weighting).  Setting β = 1.0 skips ITM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import DEFAULT_TOP_K, DATASET_DIR, DEVICE
from models.detector import YOLODetector
from models.captioner import BLIP2Captioner
from models.clip_encoder import CLIPEncoder
from scripts.index_builder import HNSWIndex
from utils.image_utils import load_image


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval pipeline
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query_img:    Image.Image,
    index:        HNSWIndex,
    clip_enc:     CLIPEncoder,
    detector:     YOLODetector,
    captioner:    Optional[BLIP2Captioner] = None,
    top_k:        int   = DEFAULT_TOP_K,
    rerank_top_k: int   = 50,       # number of candidates to re-rank
    beta:         float = 0.5,      # cosine vs ITM blend (1.0 = no ITM)
    use_reranking: bool = True,
) -> List[Dict]:
    """
    Full online retrieval pipeline.

    Parameters
    ----------
    query_img    : Raw PIL image (uncropped) from the user.
    index        : Loaded HNSW index.
    clip_enc     : Loaded CLIPEncoder.
    detector     : Loaded YOLODetector.
    captioner    : Loaded BLIP2Captioner (None → skip re-ranking).
    top_k        : Number of results to return to the user.
    rerank_top_k : Retrieve this many candidates before re-ranking.
    beta         : Weight for cosine score in blended final score.
    use_reranking: If False, return ANN results directly (faster).

    Returns
    -------
    List of result dicts sorted by final_score descending:
        {item_id, path, caption, cosine_score, itm_score, final_score}
    """
    # Step 1: YOLO crop
    cropped, crop_box = detector.crop_product(query_img)

    # Step 2: CLIP visual embedding
    query_emb = clip_enc.encode_image(cropped)        # (D,) normalised

    # Step 3: ANN retrieval
    fetch_k = rerank_top_k if (use_reranking and captioner) else top_k
    candidates = index.search(query_emb, top_k=fetch_k)

    if not use_reranking or captioner is None or not candidates:
        return candidates[:top_k]

    # Step 4: BLIP-2 ITM re-ranking
    captions = [c["caption"] for c in candidates]
    itm_scores = captioner.itm_scores_batch(cropped, captions)

    for cand, itm in zip(candidates, itm_scores):
        cosine = cand["score"]
        final  = beta * cosine + (1.0 - beta) * itm
        cand["itm_score"]   = itm
        cand["final_score"] = final

    # Sort by final_score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: encode query only (used in batch evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def encode_query(
    query_img: Image.Image,
    clip_enc:  CLIPEncoder,
    detector:  YOLODetector,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Crop + encode a query image. Returns (embedding, cropped_image).
    Used in batch evaluation scripts to separate encoding from ANN lookup.
    """
    cropped, _ = detector.crop_product(query_img)
    emb = clip_enc.encode_image(cropped)
    return emb, cropped


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class (holds loaded models — avoids reloading per query)
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Convenience class that bundles all components of the retrieval pipeline.
    Instantiate once; call `query()` repeatedly.

    Parameters
    ----------
    index_path    : Path to saved HNSW binary index.
    metadata_path : Path to saved metadata pickle.
    alpha         : Fusion weight used when the index was built.
    use_reranking : Enable BLIP-2 ITM re-ranking.
    beta          : Cosine/ITM blend weight.
    """

    def __init__(
        self,
        index_path:    Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        alpha:         float          = 0.6,
        use_reranking: bool           = True,
        beta:          float          = 0.5,
        device:        str            = DEVICE,
    ) -> None:
        from config import HNSW_INDEX_PATH, HNSW_METADATA_PATH

        print("[Pipeline] Loading models ...")
        self.detector  = YOLODetector(device=device)
        self.clip_enc  = CLIPEncoder(alpha=alpha, device=device)
        self.captioner = BLIP2Captioner(device=device) if use_reranking else None
        self.use_reranking = use_reranking
        self.beta      = beta

        idx_path  = Path(index_path  or HNSW_INDEX_PATH)
        meta_path = Path(metadata_path or HNSW_METADATA_PATH)
        self.index = HNSWIndex.load(idx_path, meta_path)

        if self.index is None:
            raise FileNotFoundError(
                f"HNSW index not found at {idx_path}. "
                "Run scripts/run_indexing.py first."
            )
        print("[Pipeline] Ready.")

    def query(
        self,
        img:    Image.Image,
        top_k:  int = DEFAULT_TOP_K,
    ) -> Tuple[List[Dict], Image.Image]:
        """
        Run the full pipeline on a PIL image.

        Returns (results_list, cropped_query_image).
        """
        results = retrieve(
            query_img    = img,
            index        = self.index,
            clip_enc     = self.clip_enc,
            detector     = self.detector,
            captioner    = self.captioner,
            top_k        = top_k,
            use_reranking= self.use_reranking,
            beta         = self.beta,
        )
        # Get the cropped image separately for display
        cropped, _ = self.detector.crop_product(img)
        return results, cropped

    def query_from_path(
        self,
        img_path: str | Path,
        top_k:    int = DEFAULT_TOP_K,
    ) -> Tuple[List[Dict], Image.Image]:
        """Convenience: load image from disk and query."""
        img = load_image(img_path)
        return self.query(img, top_k=top_k)
