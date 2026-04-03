import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_model.pth"
SPLIT_PATH = "splits.csv"
BATCH_SIZE = 64
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(SPLIT_PATH)
df["item_id"] = df["item_id"].astype(str).str.strip()

query_df = df[df["split"] == "query"].reset_index(drop=True)
gallery_df = df[df["split"] == "gallery"].reset_index(drop=True)

print("Query:", len(query_df), "| Gallery:", len(gallery_df))

# =========================
# DATASET
# =========================
class FashionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "item_id": str(row["item_id"]).strip(),
            "path": row["image_path"]
        }

query_loader = DataLoader(FashionDataset(query_df), batch_size=BATCH_SIZE)
gallery_loader = DataLoader(FashionDataset(gallery_df), batch_size=BATCH_SIZE)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# =========================
# EXTRACT EMBEDDINGS
# =========================
def extract_embeddings(loader):
    embeddings, ids, paths = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["pixel_values"].to(device)

            feats = model.vision_model(pixel_values=x).pooler_output
            feats = F.normalize(feats, dim=1)

            embeddings.append(feats.cpu())
            ids.extend(batch["item_id"])
            paths.extend(batch["path"])

    return torch.cat(embeddings), ids, paths

print("Extracting gallery embeddings...")
gallery_emb, gallery_ids, gallery_paths = extract_embeddings(gallery_loader)

print("Extracting query embeddings...")
query_emb, query_ids, query_paths = extract_embeddings(query_loader)

# =========================
# VISUALIZATION FUNCTION
# =========================
def visualize(query_idx):
    q_vec = query_emb[query_idx]
    q_id = query_ids[query_idx]

    sims = torch.matmul(q_vec, gallery_emb.T)
    topk_vals, topk_idx = torch.topk(sims, TOP_K)

    plt.figure(figsize=(15, 4))

    # Query image
    plt.subplot(1, TOP_K + 1, 1)
    img = Image.open(query_paths[query_idx])
    plt.imshow(img)
    plt.title("QUERY", fontsize=12, color="blue")
    plt.axis("off")

    # Retrieved images
    for i, idx in enumerate(topk_idx):
        plt.subplot(1, TOP_K + 1, i + 2)

        img = Image.open(gallery_paths[idx])
        plt.imshow(img)

        # Check correctness
        correct = gallery_ids[idx] == q_id
        color = "green" if correct else "red"

        score = topk_vals[i].item()

        plt.title(
            f"Top {i+1}\n{score:.2f}",
            color=color,
            fontsize=10
        )

        # Border effect
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        plt.axis("off")

    plt.tight_layout()
    plt.show()

# =========================
# RUN DEMO
# =========================
print("\nShowing sample results...\n")

for i in range(5):
    visualize(i)