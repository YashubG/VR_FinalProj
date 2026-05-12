"""
streamlit_app/app.py
--------------------
Interactive Streamlit demo for the Visual Product Search Engine.

The UI is organized as a guided studio:
1. Upload a person image
2. Let YOLO detect the garment region
3. Choose full / upper / lower body search scope
4. Optionally drag-crop manually
5. Confirm and run retrieval
6. Inspect ranked results in a gallery layout

Run with:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from PIL import Image
import streamlit as st

# Ensure project root is on sys.path regardless of where Streamlit is launched from
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import DEFAULT_TOP_K, DATASET_DIR
from utils.image_utils import resize_for_display


st.set_page_config(
    page_title="Visual Product Search Studio",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(255, 122, 24, 0.18), transparent 25%),
                radial-gradient(circle at 85% 8%, rgba(89, 114, 255, 0.15), transparent 22%),
                linear-gradient(180deg, #090d14 0%, #0b1120 44%, #0a0f17 100%);
            color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .hero {
            padding: 1.6rem 1.7rem 1.35rem 1.7rem;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 26px 80px rgba(0,0,0,0.28);
            backdrop-filter: blur(18px);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.45rem;
            line-height: 1.05;
            color: #ffffff;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0.8rem 0 0 0;
            color: rgba(240,244,252,0.84);
            font-size: 1.02rem;
            max-width: 74ch;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #f4f7ff;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .panel {
            background: rgba(7, 10, 17, 0.72);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 24px;
            padding: 1.05rem 1.08rem;
            box-shadow: 0 20px 50px rgba(0,0,0,0.20);
            backdrop-filter: blur(14px);
        }

        .panel-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            color: #9aa6bf;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .panel-heading {
            font-size: 1.02rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.2rem;
        }

        .panel-subtle {
            color: #a7b0c3;
            font-size: 0.93rem;
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px;
            padding: 0.65rem 0.85rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.16);
        }

        div[data-testid="stMetricLabel"] {
            color: #b7c0d7 !important;
        }

        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.09);
            background: rgba(255,255,255,0.03);
        }

        .result-rank {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(255,122,24,0.20), rgba(255,184,77,0.16));
            color: #fff;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-bottom: 0.6rem;
        }

        .footer-note {
            color: #9aa6bf;
            font-size: 0.90rem;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_detector():
    from models.detector import YOLODetector

    return YOLODetector()


@st.cache_resource(show_spinner="Loading CLIP encoder...")
def load_clip(alpha: float):
    from models.clip_encoder import CLIPEncoder

    return CLIPEncoder(alpha=alpha)


@st.cache_resource(show_spinner="Loading BLIP-2 ITM scorer...")
def load_itm_scorer():
    from models.captioner import BLIP2ITM

    return BLIP2ITM()


@st.cache_resource(show_spinner="Loading HNSW index...")
def load_index(tag: str):
    from config import EMBEDDINGS_DIR
    from scripts.index_builder import HNSWIndex

    idx_path = EMBEDDINGS_DIR / f"{tag}_hnsw.bin"
    meta_path = EMBEDDINGS_DIR / f"{tag}_metadata.pkl"
    return HNSWIndex.load(idx_path, meta_path)


def display_result_card(result: dict, rank: int, root: Path) -> None:
    score_key = "final_score" if "final_score" in result else "score"
    score = result.get(score_key, 0.0)
    item_id = result.get("item_id", "?")
    img_path = root / result.get("path", "")

    with st.container(border=True):
        st.markdown(
            f'<div class="result-rank">#{rank} &nbsp; Score {score:.3f} &nbsp; Item {item_id}</div>',
            unsafe_allow_html=True,
        )
        if img_path.exists():
            img = Image.open(str(img_path)).convert("RGB")
            st.image(resize_for_display(img, 360), use_container_width=True)
        else:
            st.warning("Image not found on disk.")

        cols = st.columns(2)
        with cols[0]:
            st.metric("Cosine", f"{result.get('score', 0.0):.3f}")
        with cols[1]:
            if "itm_score" in result:
                st.metric("ITM", f"{result['itm_score']:.3f}")
            else:
                st.metric("ITM", "--")

        caption = result.get("caption", "")
        if caption:
            with st.expander("Caption"):
                st.write(caption)


def make_body_variant(img: Image.Image, mode: str) -> Image.Image:
    w, h = img.size
    if mode == "Full body":
        return img
    if mode == "Upper body":
        return img.crop((0, 0, w, int(h * 0.62)))
    if mode == "Lower body":
        return img.crop((0, int(h * 0.38), w, h))
    return img


def manual_adjustment_widget(base_img: Image.Image) -> Image.Image:
    st.markdown('<div class="panel-title">Manual refinement</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtle">Use the crop box to drag and resize the garment region. '
        'If the cropper is unavailable, a slider fallback is shown instead.</div>',
        unsafe_allow_html=True,
    )

    manual = st.checkbox("Enable manual crop adjustment", value=False)
    if not manual:
        return base_img

    try:
        from streamlit_cropper import st_cropper

        st.caption("Drag the box, resize it, and keep the crop that best matches the garment.")
        try:
            adjusted = st_cropper(
                img_file=base_img,
                realtime_update=True,
                box_color="#ff7a18",
                return_type="image",
            )
        except TypeError:
            adjusted = st_cropper(
                base_img,
                realtime_update=True,
                box_color="#ff7a18",
                return_type="image",
            )

        if adjusted is not None:
            st.success("Manual crop captured.")
            st.image(resize_for_display(adjusted, 360), use_container_width=True)
            return adjusted

        return base_img

    except ImportError:
        st.warning("Interactive cropper is unavailable. Using slider-based crop fallback.")
    except Exception as exc:
        st.warning(f"Cropper error: {exc}. Using slider fallback.")

    w, h = base_img.size
    col_left, col_right = st.columns(2)
    with col_left:
        left = st.slider("Left boundary (%)", 0, 100, 0, 1)
        right = st.slider("Right boundary (%)", 0, 100, 100, 1)
    with col_right:
        top = st.slider("Top boundary (%)", 0, 100, 0, 1)
        bottom = st.slider("Bottom boundary (%)", 0, 100, 100, 1)

    x1 = int(w * left / 100)
    x2 = int(w * right / 100)
    y1 = int(h * top / 100)
    y2 = int(h * bottom / 100)
    adjusted = base_img.crop((x1, y1, x2, y2))
    st.image(resize_for_display(adjusted, 360), use_container_width=True)
    return adjusted


with st.sidebar:
    st.markdown('<div class="panel-title">Search configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Tuning controls</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subtle">Use the default tag for the main demo index, or switch to another tag if you built a different ablation index.</div>',
        unsafe_allow_html=True,
    )

    index_tag = st.text_input("Index tag", value="gallery", help="Must match run_indexing.py")
    alpha = st.slider("Fusion alpha (image weight)", 0.0, 1.0, 0.6, 0.05)
    top_k = st.slider("Top-K results", 1, 30, DEFAULT_TOP_K)
    use_rerank = st.checkbox("Use BLIP-2 reranking", value=False)
    padding = st.slider("YOLO crop padding", 0.0, 0.3, 0.05, 0.01)

    st.markdown("---")
    root_str = st.text_input("Dataset root", value=str(DATASET_DIR))
    st.markdown(
        '<div class="panel-subtle">Recommended run order:<br>1. build index<br>2. fine-tune CLIP if needed<br>3. launch Streamlit</div>',
        unsafe_allow_html=True,
    )

root = Path(root_str)


st.markdown(
    """
    <div class="hero">
        <h1>Visual Product Search Studio</h1>
        <p>
            Upload a person image, let YOLO isolate the garment region, choose
            full body, upper body, or lower body search scope, optionally fine-tune
            the crop with a drag box, and then search the catalogue.
        </p>
        <div class="pill-row">
            <span class="pill">YOLO crop</span>
            <span class="pill">Manual drag crop</span>
            <span class="pill">Upper / lower / full body</span>
            <span class="pill">CLIP + HNSW retrieval</span>
            <span class="pill">Optional BLIP-2 rerank</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="panel"><div class="panel-title">Input studio</div><div class="panel-heading">Upload a person image</div><div class="panel-subtle">The app will detect the clothing region, let you choose the search scope, and show a final crop before retrieval.</div></div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Drop a clothing image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload an image to start the retrieval pipeline.")
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">How it works</div>
            <div class="panel-subtle">
                1. YOLO finds the garment region.<br>
                2. Pick full, upper, or lower body search.<br>
                3. Optionally drag the crop manually.<br>
                4. Confirm the crop and run retrieval.<br>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

query_img = Image.open(uploaded).convert("RGB")

# Original image preview (stacked vertically)
with st.container(border=True):
    st.markdown('<div class="panel-title">Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Original image</div>', unsafe_allow_html=True)
    st.image(resize_for_display(query_img, 300), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# YOLO detection (Step 1)
with st.container(border=True):
    st.markdown('<div class="panel-title">Step 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">YOLO detection</div>', unsafe_allow_html=True)

    with st.spinner("Running YOLO detection..."):
        detector = load_detector()
        all_crops = detector.crop_all(query_img, padding=padding)

    detection_count = len(all_crops)
    if detection_count == 0:
        st.warning("No product region was detected, so the full image will be used as the base crop.")
        selected_detection = query_img
        selected_conf = None
    else:
        labels = [f"Region {i+1} | conf {conf:.2f}" for i, (_, _, conf) in enumerate(all_crops)]
        selected_label = st.selectbox("Detected regions", labels, index=0)
        selected_idx = labels.index(selected_label)
        selected_detection, _, selected_conf = all_crops[selected_idx]

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Detections", detection_count)
    with summary_cols[1]:
        st.metric("Padding", f"{padding:.2f}")
    with summary_cols[2]:
        st.metric("Mode", "Ready")

    if selected_conf is not None:
        st.caption(f"Selected detection confidence: {selected_conf:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# Choose search target (Step 2)
with st.container(border=True):
    st.markdown('<div class="panel-title">Step 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Choose search target</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtle">This controls whether the search is aimed at the full outfit, the upper garment region, or the lower garment region.</div>', unsafe_allow_html=True)
    target_mode = st.radio(
        "Search for",
        ["Full body", "Upper body", "Lower body"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

    base_crop = make_body_variant(selected_detection, target_mode)
    st.image(resize_for_display(base_crop, 300), use_container_width=True)
    st.markdown(
        f'<div class="panel-subtle">Preview crop: <strong>{target_mode}</strong> search scope.</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Manual crop refinement (Step 3)
with st.container(border=True):
    st.markdown('<div class="panel-title">Step 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Manual crop refinement</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtle">If the auto crop is not perfect, refine it with drag and resize controls like a phone screenshot editor.</div>', unsafe_allow_html=True)
    crop_preview = manual_adjustment_widget(base_crop)

    force_full = st.checkbox("Ignore crop and use the full uploaded image", value=False)
    final_crop = query_img if force_full else crop_preview

    st.markdown(
        f'<div class="panel-subtle">Final search image will use <strong>{"full image" if force_full else "selected crop"}</strong>.</div>',
        unsafe_allow_html=True,
    )
    st.image(resize_for_display(final_crop, 300), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Confirm and search (Step 4)
with st.container(border=True):
    st.markdown('<div class="panel-title">Step 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Confirm and search</div>', unsafe_allow_html=True)
    search_now = st.button("Run retrieval", type="primary", use_container_width=True)

if not search_now:
    st.markdown(
        '<div class="panel"><div class="panel-subtle">Adjust the crop, choose the garment region, and press <strong>Run retrieval</strong> when ready.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="panel-title">Retrieval engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Loading index and encoders</div>', unsafe_allow_html=True)

    index = load_index(index_tag)
    if index is None:
        st.error(
            f"Index '{index_tag}' was not found in embeddings/. Build it first using run_indexing.py."
        )
        st.stop()

    st.success(f"Index loaded with {len(index):,} vectors.")

    clip_enc = load_clip(alpha)
    with st.spinner("Encoding query crop with CLIP..."):
        query_emb = clip_enc.encode_image(final_crop)

    with st.spinner(f"Searching HNSW (top-{top_k})..."):
        rerank_k = top_k * 5
        candidates = index.search(query_emb, top_k=rerank_k if use_rerank else top_k)

    if use_rerank and candidates:
        with st.spinner("BLIP-2 ITM reranking..."):
            itm_scorer = load_itm_scorer()
            captions   = [c["caption"] for c in candidates]
            itm_scores = itm_scorer.itm_scores_batch(final_crop, captions)
            for cand, itm in zip(candidates, itm_scores):
                cand["itm_score"]   = itm
                cand["final_score"] = 0.5 * cand["score"] + 0.5 * itm
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            candidates = candidates[:top_k]


st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-title">Results gallery</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-heading">Top retrieved products</div>', unsafe_allow_html=True)

if not candidates:
    st.warning("No retrieval candidates were returned.")
    st.stop()

st.markdown(
    f'<div class="panel-subtle">Showing top {min(top_k, len(candidates))} matches for <strong>{target_mode}</strong> search.</div>',
    unsafe_allow_html=True,
)

grid_cols = 3
for start in range(0, min(top_k, len(candidates)), grid_cols):
    cols = st.columns(grid_cols, gap="large")
    for offset, col in enumerate(cols):
        idx = start + offset
        if idx >= min(top_k, len(candidates)):
            continue
        with col:
            display_result_card(candidates[idx], rank=idx + 1, root=root)

with st.expander("Raw result data"):
    import pandas as pd

    rows: List[dict] = []
    for i, r in enumerate(candidates[:top_k]):
        rows.append(
            {
                "Rank": i + 1,
                "Item ID": r.get("item_id", ""),
                "Path": r.get("path", ""),
                "Cosine": f"{r.get('score', 0):.4f}",
                "ITM": f"{r.get('itm_score', ''):.4f}" if "itm_score" in r else "—",
                "Final": f"{r.get('final_score', r.get('score', 0)):.4f}",
                "Caption": (r.get("caption", "")[:60] + "...") if r.get("caption") else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.markdown(
    '<div class="footer-note">Visual Product Search Studio · DeepFashion In-Shop Clothes Retrieval</div>',
    unsafe_allow_html=True,
)