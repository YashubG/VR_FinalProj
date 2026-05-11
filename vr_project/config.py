"""
config.py
---------
Central configuration for the Visual Product Search Engine.

All hyperparameters, model names, paths, and runtime flags live here.
Changing a single value in this file propagates everywhere.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project root  (this file's parent directory)
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────
DATA_DIR         = ROOT_DIR / "data"
DATASET_DIR      = DATA_DIR / "deepfashion"          # raw images
SPLIT_DIR        = DATA_DIR / "splits"               # train/query/gallery lists
EMBEDDINGS_DIR   = ROOT_DIR / "embeddings"           # saved numpy arrays
MODELS_DIR       = ROOT_DIR / "models"               # local model checkpoints
RESULTS_DIR      = ROOT_DIR / "results"

for d in [DATA_DIR, DATASET_DIR, SPLIT_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Model identifiers
# (HuggingFace / OpenCLIP hub names used when
#  local checkpoints are not present)
# ─────────────────────────────────────────────
CLIP_MODEL_NAME        = "ViT-B/32"           # open_clip arch
CLIP_PRETRAINED        = "openai"             # open_clip weights tag
CLIP_LOCAL_PATH        = MODELS_DIR / "clip_finetuned.pt"   # saved after fine-tuning
CLIP_CHECKPOINT_DIR    = MODELS_DIR / "clip_checkpoints"

BLIP2_MODEL_NAME       = "Salesforce/blip2-opt-2.7b"   # HuggingFace id
BLIP2_LOCAL_PATH       = MODELS_DIR / "blip2"          # local save dir

YOLO_MODEL_NAME        = "yolov8n.pt"          # ultralytics auto-download
YOLO_LOCAL_PATH        = MODELS_DIR / "yolo" / "yolov8n.pt"

# ─────────────────────────────────────────────
# HNSW index
# ─────────────────────────────────────────────
HNSW_INDEX_PATH        = EMBEDDINGS_DIR / "hnsw_index.bin"
HNSW_METADATA_PATH     = EMBEDDINGS_DIR / "metadata.pkl"
HNSW_M                 = 32       # number of bidirectional links per node
HNSW_EF_CONSTRUCTION   = 200     # index-build quality
HNSW_EF_SEARCH         = 100     # search-time exploration factor
EMBEDDING_DIM          = 512      # CLIP ViT-B/32 output dim

# ─────────────────────────────────────────────
# Fusion weight
# ─────────────────────────────────────────────
ALPHA                  = 0.6      # image weight; (1-ALPHA) is text weight
                                  # ablation experiments override this

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE             = 8
NUM_EPOCHS             = 10
LEARNING_RATE          = 1e-5
WEIGHT_DECAY           = 1e-4
TEMPERATURE            = 0.07     # contrastive loss temperature
TRAIN_LAST_N_BLOCKS    = 4        # fine-tune only last N vision transformer blocks
SEEDS                  = [42, 0, 7, 21]  # stand-ins for roll-number seeds

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
TOP_K_VALUES           = [5, 10, 15]  # evaluation K values
DEFAULT_TOP_K          = 10

# ─────────────────────────────────────────────
# YOLO detection
# ─────────────────────────────────────────────
YOLO_CONF_THRESHOLD    = 0.25
YOLO_IOU_THRESHOLD     = 0.45
YOLO_IMAGE_SIZE        = 640

# ─────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────
IMAGE_SIZE             = 224      # CLIP input resolution
PIXEL_MEAN             = (0.48145466, 0.4578275,  0.40821073)
PIXEL_STD              = (0.26862954, 0.26130258, 0.27577711)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
