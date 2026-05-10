"""
models/detector.py
------------------
YOLO-based product localisation module.

Design decisions
----------------
* We use Ultralytics YOLOv8 (model-agnostic: any YOLO variant works).
* The detector is wrapped in a class so the model is loaded once and reused.
* When a local checkpoint exists, it is loaded from disk; otherwise the
  Ultralytics auto-download fetches from the official CDN and saves locally.
* Only the highest-confidence detection per image is used as the product crop.
  Multiple detections are returned for advanced use (e.g. re-crop UI).
* If no detection is found the full image is returned as fallback — this
  avoids hard failures on clean studio shots where the whole frame is the
  product.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from config import (
    YOLO_MODEL_NAME,
    YOLO_LOCAL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_IMAGE_SIZE,
    DEVICE,
)
from utils.image_utils import crop_box


# ─────────────────────────────────────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────────────────────────────────────
Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels


# ─────────────────────────────────────────────────────────────────────────────
# Helper: save YOLO model
# ─────────────────────────────────────────────────────────────────────────────

def save_yolo_model(model, save_path: Path) -> None:
    """
    Copy the YOLO checkpoint to a local path so future runs don't need internet.

    Ultralytics YOLO models save themselves as .pt files; we just copy the
    already-downloaded weights from the ultralytics cache to our MODELS_DIR.
    """
    import shutil
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # model.ckpt_path is the path ultralytics used internally
    src = Path(model.ckpt_path)
    if src.resolve() != save_path.resolve():
        shutil.copy2(str(src), str(save_path))
    print(f"[Detector] YOLO weights saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wrapper around a YOLO model for single-product localisation.

    Parameters
    ----------
    local_path  : Path to a locally saved .pt checkpoint.
                  If the file exists it is loaded directly (no internet needed).
    model_name  : Ultralytics model identifier used for auto-download when
                  local_path does not exist.
    conf        : Minimum confidence threshold for detections.
    iou         : NMS IoU threshold.
    imgsz       : Inference image size (pixels, square).
    device      : 'cuda' or 'cpu'.
    save_after_load : If True AND we just auto-downloaded, save to local_path.
    """

    def __init__(
        self,
        local_path:       Path   = YOLO_LOCAL_PATH,
        model_name:       str    = YOLO_MODEL_NAME,
        conf:             float  = YOLO_CONF_THRESHOLD,
        iou:              float  = YOLO_IOU_THRESHOLD,
        imgsz:            int    = YOLO_IMAGE_SIZE,
        device:           str    = DEVICE,
        save_after_load:  bool   = True,
    ) -> None:
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device

        self._model = self._load(local_path, model_name, save_after_load)

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(
        self,
        local_path:      Path,
        model_name:      str,
        save_after_load: bool,
    ):
        """Load from local disk if available, otherwise download and optionally save."""
        from ultralytics import YOLO

        local_path = Path(local_path)
        if local_path.exists():
            print(f"[Detector] Loading YOLO from local path: {local_path}")
            model = YOLO(str(local_path))
        else:
            print(f"[Detector] Local checkpoint not found. Downloading '{model_name}' ...")
            model = YOLO(model_name)  # auto-downloads to ultralytics cache
            if save_after_load:
                save_yolo_model(model, local_path)

        return model

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(
        self,
        img: Image.Image,
    ) -> List[Tuple[Box, float, int]]:
        """
        Run YOLO on a PIL image.

        Returns
        -------
        List of (box, confidence, class_id) sorted by confidence descending.
        box is (x1, y1, x2, y2) in pixel space.
        Empty list if no detection passes the confidence threshold.
        """
        results = self._model(
            img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Tuple[Box, float, int]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # (N, 4)
            confs       = result.boxes.conf.cpu().numpy()   # (N,)
            classes     = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            for box, conf_score, cls_id in zip(boxes_xyxy, confs, classes):
                detections.append((tuple(box.tolist()), float(conf_score), int(cls_id)))

        # Sort by confidence descending
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    def crop_product(
        self,
        img:     Image.Image,
        padding: float = 0.05,
    ) -> Tuple[Image.Image, Optional[Box]]:
        """
        Return the cropped product image (highest-confidence detection)
        and the raw bounding box.

        Falls back to the full image if YOLO finds nothing.
        """
        detections = self.detect(img)
        if not detections:
            return img, None
        best_box, _, _ = detections[0]
        cropped = crop_box(img, best_box, padding=padding)
        return cropped, best_box

    def crop_all(
        self,
        img:     Image.Image,
        padding: float = 0.05,
    ) -> List[Tuple[Image.Image, Box, float]]:
        """
        Return all detected crops (used for re-crop UI in Streamlit).
        Each entry is (cropped_image, box, confidence).
        """
        detections = self.detect(img)
        results = []
        for box, conf, _ in detections:
            crop = crop_box(img, box, padding=padding)
            results.append((crop, box, conf))
        return results
