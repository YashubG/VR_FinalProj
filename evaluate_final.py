import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import os
from huggingface_hub import login

def safe_login():
    token = os.getenv("HF_TOKEN")

    if not token:
        try:
            with open("Hugging_Face_Token.txt") as f:
                token = f.read().strip()
        except FileNotFoundError:
            pass

    if token:
        login(token)
    else:
        print("Running without Hugging Face login.")

safe_login()
# =========================
# CONFIG
# =========================
DATA_DIR = "Data/img"
MODEL_PATH = "best_model.pth"
SPLIT_PATH = "splits.csv"
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD SPLITS (CRITICAL FIX)
# =========================
df = pd.read_csv(SPLIT_PATH)

# remove any hidden whitespace
df["item_id"] = df["item_id"].astype(str).str.strip()

query_df = df[df["split"] == "query"].reset_index(drop=True)
gallery_df = df[df["split"] == "gallery"].reset_index(drop=True)

print("Query size:", len(query_df))
print("Gallery size:", len(gallery_df))

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
            "item_id": str(row["item_id"]).strip()
        }

query_loader = DataLoader(FashionDataset(query_df), batch_size=BATCH_SIZE)
gallery_loader = DataLoader(FashionDataset(gallery_df), batch_size=BATCH_SIZE)

# =========================
# LOAD MODEL
# =========================
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# =========================
# EMBEDDINGS
# =========================
def extract_embeddings(loader):
    embeddings, ids = [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["pixel_values"].to(device)

            feats = model.vision_model(pixel_values=x).pooler_output
            feats = F.normalize(feats, dim=1)

            embeddings.append(feats.cpu())
            ids.extend([str(i).strip() for i in batch["item_id"]])

    return torch.cat(embeddings), ids

print("\nExtracting gallery embeddings...")
gallery_emb, gallery_ids = extract_embeddings(gallery_loader)

print("\nExtracting query embeddings...")
query_emb, query_ids = extract_embeddings(query_loader)

# =========================
# DEBUG CHECK (VERY IMPORTANT)
# =========================
query_set = set(query_ids)
gallery_set = set(gallery_ids)

print("\nDEBUG INFO:")
print("Unique query IDs:", len(query_set))
print("Unique gallery IDs:", len(gallery_set))
print("Overlap:", len(query_set & gallery_set))

if len(query_set & gallery_set) == 0:
    print("❌ ERROR: No overlap → metrics will be zero!")
    exit()

# =========================
# METRICS
# =========================
def compute_metrics(query_emb, gallery_emb, query_ids, gallery_ids, K_values=[5,10,15]):
    results = {}

    for K in K_values:
        recall_list, ap_list, ndcg_list = [], [], []

        for i in range(len(query_emb)):
            sims = torch.matmul(query_emb[i], gallery_emb.T)
            topk = torch.topk(sims, K).indices

            retrieved_ids = [gallery_ids[j] for j in topk]
            rel = np.array([1 if rid == query_ids[i] else 0 for rid in retrieved_ids])

            # Recall@K
            recall_list.append(1 if rel.sum() > 0 else 0)

            # mAP@K
            precisions = []
            correct = 0
            for idx, r in enumerate(rel):
                if r:
                    correct += 1
                    precisions.append(correct / (idx + 1))
            ap_list.append(np.mean(precisions) if precisions else 0)

            # NDCG@K
            dcg = sum(rel[j] / np.log2(j+2) for j in range(len(rel)))
            ideal = sorted(rel, reverse=True)
            idcg = sum(ideal[j] / np.log2(j+2) for j in range(len(rel)))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0)

        results[K] = {
            "Recall": np.mean(recall_list),
            "mAP": np.mean(ap_list),
            "NDCG": np.mean(ndcg_list)
        }

    return results

# =========================
# RUN EVALUATION
# =========================
metrics = compute_metrics(query_emb, gallery_emb, query_ids, gallery_ids)

print("\n===== FINAL METRICS =====")
for K, vals in metrics.items():
    print(f"\n@{K}")
    print(f"Recall@{K}: {vals['Recall']:.4f}")
    print(f"mAP@{K}:    {vals['mAP']:.4f}")
    print(f"NDCG@{K}:   {vals['NDCG']:.4f}")