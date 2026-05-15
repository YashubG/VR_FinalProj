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


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_detector():
    from models.detector import YOLODetector

    return YOLODetector()


@st.cache_resource(show_spinner="Loading CLIP encoder...")
def load_clip(alpha: float, use_finetuned: bool, checkpoint_path: Path | None = None):
    from models.clip_encoder import CLIPEncoder

    if use_finetuned:
        if checkpoint_path is None:
            from config import CLIP_LOCAL_PATH

            checkpoint_path = CLIP_LOCAL_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Fine-tuned CLIP checkpoint not found: {checkpoint_path}")
        return CLIPEncoder(alpha=alpha, use_finetuned=True, local_finetuned_path=checkpoint_path)

    return CLIPEncoder(alpha=alpha, use_finetuned=False)


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


def parse_ablation_name(filename: str) -> tuple[str, float, int | None]:
    """
    Parse ablation filename to extract:
      - ablation type (A, B, C, ...)
      - alpha value (e.g., 0.6 from '_a06')
      - seed (e.g., 116 from '_seed_116')
    
    Examples:
      ablation_A_hnsw.bin -> ("A", 1.0, None)
      ablation_B_a06_hnsw.bin -> ("B", 0.6, None)
      ablation_C_a06_seed_116_hnsw.bin -> ("C", 0.6, 116)
    """
    base = filename.replace("_hnsw.bin", "").replace("_metadata.pkl", "")
    parts = base.split("_")
    
    if not parts[0] == "ablation" or len(parts) < 2:
        return None
    
    ablation_type = parts[1]  # A, B, C, ...
    alpha = 1.0
    seed = None
    
    for i, part in enumerate(parts[2:], start=2):
        if part.startswith("a") and len(part) >= 3:
            # _a06 -> 0.6, _a08 -> 0.8
            try:
                digits = part[1:]  # Remove 'a' prefix
                alpha = int(digits) / 10.0  # Two digits: 06 -> 0.6, 08 -> 0.8
            except ValueError:
                pass
        elif part == "seed" and len(parts) > i + 1:
            try:
                seed = int(parts[i + 1])
            except (ValueError, IndexError):
                pass
    
    return (ablation_type, alpha, seed)


def get_available_ablations() -> list[tuple[str, float, int | None, str]]:
    """
    Scan embeddings folder for available ablations.
    Returns list of (ablation_name, alpha, seed, display_name)
    """
    from config import EMBEDDINGS_DIR
    
    ablations = {}
    
    if not EMBEDDINGS_DIR.exists():
        return []
    
    for f in EMBEDDINGS_DIR.glob("ablation_*_hnsw.bin"):
        parsed = parse_ablation_name(f.name)
        if parsed:
            abl_type, alpha, seed = parsed
            key = (abl_type, alpha, seed)
            if key not in ablations:
                # Build display name
                seed_str = f" (seed {seed})" if seed else ""
                ablations[key] = f"Ablation {abl_type} | α={alpha:.2f}{seed_str}"
    
    return sorted(ablations.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0))


def get_ablation_settings(abl_type: str, alpha: float) -> dict:
    """Infer model settings based on ablation type and alpha."""
    if abl_type == "A":
        return {
            "use_finetuned": False,
            "use_rerank": False,
            "caption": "Vision-only CLIP baseline (no text, no ITM)",
        }
    if abl_type == "B":
        return {
            "use_finetuned": False,
            "use_rerank": True,
            "caption": "Frozen CLIP + frozen BLIP-2 (caption fusion enabled)",
        }
    if abl_type == "C":
        return {
            "use_finetuned": True,
            "use_rerank": True,
            "caption": "Fine-tuned CLIP + frozen BLIP-2 (best-performing demo mode)",
        }
    # Default fallback
    return {
        "use_finetuned": False,
        "use_rerank": False,
        "caption": f"Ablation {abl_type} (default settings)",
    }


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

        # Caption removed: only show Cosine and ITM per design


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

    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))

    x1 = int(w * left / 100)
    x2 = int(w * right / 100)
    y1 = int(h * top / 100)
    y2 = int(h * bottom / 100)

    if x2 <= x1 or y2 <= y1:
        st.warning("The selected crop is empty. Using the base crop instead.")
        return base_img

    adjusted = base_img.crop((x1, y1, x2, y2))
    st.image(resize_for_display(adjusted, 360), use_container_width=True)
    return adjusted


def main():
    with st.sidebar:
        st.markdown('<div class="panel-title">Search configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-heading">Ablation selection</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtle">The app auto-detects available ablations from the embeddings folder. Select one to lock the matching settings.</div>',
            unsafe_allow_html=True,
        )

        # Auto-detect available ablations
        available = get_available_ablations()
        if not available:
            st.error("No ablations found in embeddings/ folder. Run run_indexing.py first.")
            st.stop()

        # Build dropdown options
        options_dict = {display: (abl_type, alpha, seed) for (abl_type, alpha, seed), display in available}
        selected_display = st.selectbox(
            "Available ablations",
            list(options_dict.keys()),
            label_visibility="collapsed",
            help="Auto-detected from embeddings folder filenames"
        )
        
        abl_type, alpha, seed = options_dict[selected_display]
        settings = get_ablation_settings(abl_type, alpha)
        
        # Build index tag using the filename convention from embeddings/.
        # Examples: alpha=0.6 -> _a06, alpha=0.8 -> _a08, alpha=1.0 -> no suffix.
        seed_str = f"_seed_{seed}" if seed else ""
        alpha_str = f"_a{int(round(alpha * 10)):02d}" if alpha != 1.0 else ""
        index_tag = f"ablation_{abl_type}{alpha_str}{seed_str}"

        # Advanced overrides (collapsed by default)
        with st.expander("Advanced overrides (expert mode)"):
            st.markdown('<div class="panel-subtle">Only change these if you understand the ablations.</div>', unsafe_allow_html=True)
            override_index = st.text_input("Override index tag", value=index_tag, help="Leave blank to use auto-detected tag")
            index_tag = override_index if override_index.strip() else index_tag
            
            override_alpha = st.checkbox("Override alpha", value=False)
            if override_alpha and settings["use_finetuned"]:
                alpha = st.slider("Fusion alpha (image weight)", 0.0, 1.0, alpha, 0.05)
            elif not override_alpha:
                st.slider("Fusion alpha (image weight)", 0.0, 1.0, alpha, 0.05, disabled=True)

        top_k = st.slider("Top-K results", 1, 30, DEFAULT_TOP_K)
        padding = st.slider("YOLO crop padding", 0.0, 0.3, 0.05, 0.01)

        st.markdown("---")
        root_str = st.text_input("Dataset root", value=str(DATASET_DIR))
        st.markdown(
            f'<div class="panel-subtle"><strong>{settings["caption"]}</strong><br>✓ Index: <code>{index_tag}</code><br>✓ α = {alpha:.2f}<br>✓ ITM reranking: {"enabled" if settings["use_rerank"] else "disabled"}</div>',
            unsafe_allow_html=True,
        )

    root = Path(root_str)
    use_rerank = settings["use_rerank"]
    use_finetuned = settings["use_finetuned"]


    st.markdown(
        """
        <div class="hero">
            <h1>Visual Product Search Studio</h1>
            <p>
                Upload a person image, let YOLO isolate the garment region, choose
                full body, upper body, or lower body search scope, optionally refine
                the crop, confirm it, and then search the catalogue.
            </p>
            <div class="pill-row">
                <span class="pill">YOLO crop</span>
                <span class="pill">Manual drag crop</span>
                <span class="pill">Upper / lower / full body</span>
                <span class="pill">CLIP + HNSW retrieval</span>
                <span class="pill">Ablation-aware demo</span>
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
                    2. Choose full / upper / lower body scope.<br>
                    3. Optionally refine the crop manually.<br>
                    4. Confirm the crop before retrieval.<br>
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

    with st.container(border=True):
        st.markdown('<div class="panel-title">Step 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-heading">Choose search scope</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtle">Pick the garment region to search: full body, upper body, or lower body. This mirrors the search-scope step in the PDF.</div>', unsafe_allow_html=True)
        target_mode = st.radio(
            "Search scope",
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

    with st.container(border=True):
        st.markdown('<div class="panel-title">Step 3</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-heading">Manual crop refinement</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtle">Refine the chosen scope only if needed, then confirm the crop before search.</div>', unsafe_allow_html=True)
        crop_preview = manual_adjustment_widget(base_crop)
        st.image(resize_for_display(crop_preview, 300), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="panel-title">Step 4</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-heading">Confirm crop</div>', unsafe_allow_html=True)
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            confirm_crop = st.button("Confirm crop", type="primary", use_container_width=True)
        with confirm_cols[1]:
            recrop = st.button("Re-crop", use_container_width=True)

        if recrop:
            st.warning("Crop adjustment requested. Change the crop or search scope, then confirm again.")
            st.stop()

        if not confirm_crop:
            st.info("Confirm the crop to continue to retrieval.")
            st.stop()

        final_crop = crop_preview
        st.markdown(
            f'<div class="panel-subtle">Confirmed search image uses <strong>{target_mode}</strong> scope with the selected crop.</div>',
            unsafe_allow_html=True,
        )
        st.image(resize_for_display(final_crop, 300), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Retrieval only happens after the crop has been confirmed.
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

        # Load CLIP with potential seed-specific checkpoint
        checkpoint_path = None
        if use_finetuned and seed:
            from config import MODELS_DIR
            seed_checkpoint = MODELS_DIR / f"clip_finetuned_seed_{seed}.pt"
            if seed_checkpoint.exists():
                checkpoint_path = seed_checkpoint
                st.info(f"Using seed-{seed} CLIP checkpoint")
        
        clip_enc = load_clip(alpha, use_finetuned, checkpoint_path)
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
        f'<div class="panel-subtle">Showing top {min(top_k, len(candidates))} matches for selected search.</div>',
        unsafe_allow_html=True,
    )

    # Only compute ITM when the selected ablation uses BLIP-2 reranking.
    need_itm = use_rerank and any("itm_score" not in c for c in candidates[:top_k])
    if need_itm and candidates:
        try:
            with st.spinner("Computing ITM scores for display..."):
                itm_scorer = load_itm_scorer()
                captions = [c.get("caption", "") for c in candidates[:top_k]]
                itm_scores = itm_scorer.itm_scores_batch(final_crop, captions)
                for c, s in zip(candidates[:top_k], itm_scores):
                    c["itm_score"] = s
        except Exception as exc:
            st.warning(f"ITM scorer unavailable or failed: {exc}")

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
                    # Caption removed from raw table — UI presents Cosine and ITM only
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="footer-note">Visual Product Search Studio · DeepFashion In-Shop Clothes Retrieval</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    inject_styles()
    main()