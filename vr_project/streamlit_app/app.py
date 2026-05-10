"""
streamlit_app/app.py
--------------------
Interactive Streamlit demo for the Visual Product Search Engine.

Flow
----
1. User uploads a query image.
2. YOLO detects and crops the product region.
3. User confirms the crop or selects a different one.
4. System runs CLIP encoding → HNSW ANN search → (optional) BLIP-2 re-ranking.
5. Top-K results are displayed with similarity scores and metadata.

Run with:
    streamlit run streamlit_app/app.py
    OR from project root:
    streamlit run streamlit_app/app.py

Design notes
------------
* Models are loaded once via @st.cache_resource (survives re-runs).
* The crop-confirmation step mirrors the spec requirement exactly.
* The app degrades gracefully: if the HNSW index is not built, it shows
  a clear error with instructions to run run_indexing.py.
* Top-K is adjustable via a sidebar slider.
* The full pipeline (including re-ranking) can be toggled for speed.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where Streamlit is launched from
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import io
import numpy as np
import streamlit as st
from PIL import Image

from config import DEFAULT_TOP_K, DATASET_DIR, DEVICE
from utils.image_utils import resize_for_display


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Visual Product Search",
    page_icon   = "👗",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders (load once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_detector():
    from models.detector import YOLODetector
    return YOLODetector()


@st.cache_resource(show_spinner="Loading CLIP encoder...")
def load_clip(alpha: float):
    from models.clip_encoder import CLIPEncoder
    return CLIPEncoder(alpha=alpha)


@st.cache_resource(show_spinner="Loading BLIP-2 captioner...")
def load_captioner():
    from models.captioner import BLIP2Captioner
    return BLIP2Captioner()


@st.cache_resource(show_spinner="Loading HNSW index...")
def load_index(tag: str):
    from scripts.index_builder import HNSWIndex
    from config import EMBEDDINGS_DIR

    idx_path  = EMBEDDINGS_DIR / f"{tag}_hnsw.bin"
    meta_path = EMBEDDINGS_DIR / f"{tag}_metadata.pkl"
    return HNSWIndex.load(idx_path, meta_path)


# ─────────────────────────────────────────────────────────────────────────────
# Result display helpers
# ─────────────────────────────────────────────────────────────────────────────

def display_result_card(result: dict, rank: int, root: Path):
    """Render one retrieved result in a Streamlit column."""
    score_key = "final_score" if "final_score" in result else "score"
    score     = result.get(score_key, 0.0)

    st.markdown(f"**#{rank}** &nbsp; `score: {score:.3f}`")
    st.caption(f"item_id: `{result.get('item_id','?')}`")

    # Try to show the actual image from disk
    img_path = root / result.get("path", "")
    if img_path.exists():
        img = Image.open(str(img_path)).convert("RGB")
        img = resize_for_display(img, max_size=200)
        st.image(img, use_container_width=True)
    else:
        st.warning("Image not found on disk.")

    caption = result.get("caption", "")
    if caption:
        with st.expander("Caption"):
            st.write(caption)

    if "itm_score" in result:
        st.metric("ITM score", f"{result['itm_score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar settings
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    index_tag   = st.text_input("Index tag", value="gallery",
                                help="Must match the tag used in run_indexing.py")
    alpha       = st.slider("Fusion α (image weight)", 0.0, 1.0, 0.6, 0.05,
                            help="α=1.0 → vision only. α=0 → text only.")
    top_k       = st.slider("Top-K results", 1, 30, DEFAULT_TOP_K)
    use_rerank  = st.checkbox("BLIP-2 re-ranking", value=False,
                              help="Improves quality but is slower")
    padding     = st.slider("YOLO crop padding", 0.0, 0.3, 0.05, 0.01)

    st.markdown("---")
    root_str    = st.text_input("Dataset root (for showing images)",
                                value=str(DATASET_DIR))
    st.markdown("---")
    st.info(
        "**Run order:**\n"
        "1. `run_indexing.py` (build index)\n"
        "2. *(optional)* `run_finetune.py`\n"
        "3. This Streamlit app"
    )


root = Path(root_str)

# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

st.title("👗 Visual Product Search Engine")
st.markdown(
    "Upload a clothing image. The system detects the product, "
    "encodes it with CLIP, and retrieves visually similar items from the index."
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("⬆️ Upload an image to begin.")
    st.stop()

query_img = Image.open(uploaded).convert("RGB")

col_orig, col_crop = st.columns(2)
with col_orig:
    st.subheader("Original image")
    st.image(resize_for_display(query_img, 400), use_container_width=True)

# ── YOLO detection ────────────────────────────────────────────────────────────
st.subheader("Step 1 — Product detection (YOLO)")

with st.spinner("Running YOLO ..."):
    detector = load_detector()
    all_crops = detector.crop_all(query_img, padding=padding)

if not all_crops:
    st.warning("No product detected. Using full image as crop.")
    cropped_img = query_img
    crop_box    = None
else:
    # Show all detected crops for user to choose
    if len(all_crops) == 1:
        cropped_img, crop_box, conf = all_crops[0]
        st.success(f"Detected 1 product region (confidence: {conf:.2f})")
    else:
        st.info(f"Detected {len(all_crops)} product regions. Select one:")
        crop_options = {
            f"Crop {i+1} (conf={c:.2f})": img
            for i, (img, _, c) in enumerate(all_crops)
        }
        selected_label = st.radio(
            "Choose crop",
            list(crop_options.keys()),
            horizontal=True,
        )
        sel_idx         = list(crop_options.keys()).index(selected_label)
        cropped_img, crop_box, conf = all_crops[sel_idx]

with col_crop:
    st.subheader("Cropped product")
    st.image(resize_for_display(cropped_img, 400), use_container_width=True)

# ── User confirmation ─────────────────────────────────────────────────────────
st.markdown("---")
confirm_col, recrop_col = st.columns(2)
with confirm_col:
    confirmed = st.button("✅ Confirm crop — run search", type="primary",
                          use_container_width=True)
with recrop_col:
    if st.button("🔄 Use full image instead", use_container_width=True):
        cropped_img = query_img
        st.session_state["use_full"] = True

if not confirmed:
    st.info("Confirm the crop above to continue.")
    st.stop()

# ── Load index ────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Loading index ...")
index = load_index(index_tag)

if index is None:
    st.error(
        f"❌ Index `{index_tag}` not found in `embeddings/`.\n\n"
        "**Run first:**\n```\npython run_indexing.py\n```"
    )
    st.stop()

st.success(f"Index loaded: **{len(index):,}** product vectors")

# ── Encode & retrieve ─────────────────────────────────────────────────────────
st.subheader("Step 3 — Retrieval")

with st.spinner("Encoding query image with CLIP ..."):
    clip_enc  = load_clip(alpha)
    query_emb = clip_enc.encode_image(cropped_img)

with st.spinner(f"Searching HNSW index (top-{top_k}) ..."):
    rerank_k  = top_k * 5
    candidates = index.search(query_emb, top_k=rerank_k if use_rerank else top_k)

if use_rerank:
    with st.spinner("BLIP-2 ITM re-ranking ..."):
        captioner = load_captioner()
        captions  = [c["caption"] for c in candidates]
        itm_scores = captioner.itm_scores_batch(cropped_img, captions)
        for cand, itm in zip(candidates, itm_scores):
            cand["itm_score"]   = itm
            cand["final_score"] = 0.5 * cand["score"] + 0.5 * itm
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        candidates = candidates[:top_k]

# ── Display results ───────────────────────────────────────────────────────────
st.subheader(f"Step 4 — Top-{top_k} Results")
st.markdown("---")

n_cols = min(5, top_k)
cols   = st.columns(n_cols)

for i, result in enumerate(candidates[:top_k]):
    with cols[i % n_cols]:
        display_result_card(result, rank=i + 1, root=root)

# ── Metadata table ────────────────────────────────────────────────────────────
with st.expander("📊 Raw result data"):
    import pandas as pd
    rows = []
    for i, r in enumerate(candidates[:top_k]):
        rows.append({
            "Rank":    i + 1,
            "Item ID": r.get("item_id", ""),
            "Path":    r.get("path",    ""),
            "Cosine":  f"{r.get('score', 0):.4f}",
            "ITM":     f"{r.get('itm_score', ''):.4f}" if "itm_score" in r else "—",
            "Final":   f"{r.get('final_score', r.get('score', 0)):.4f}",
            "Caption": r.get("caption", "")[:60] + "..." if r.get("caption") else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.caption("Visual Product Search Engine · DeepFashion In-Shop Clothes Retrieval")
