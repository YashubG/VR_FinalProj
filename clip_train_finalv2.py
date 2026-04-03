import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
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
BATCH_SIZE = 32
EPOCHS = 20
K = 4
LR = 5e-6

CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "best_model.pth"
SPLIT_PATH = "splits.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# STEP 1: BUILD DATAFRAME
# =========================
data = []

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".jpg"):
            path = os.path.join(root, file)

            item_id = None
            for p in path.split(os.sep):
                if p.startswith("id_"):
                    item_id = p
                    break

            if item_id:
                data.append([path, item_id])

df = pd.DataFrame(data, columns=["image_path", "item_id"])
print("Total images:", len(df))

# =========================
# STEP 2: CORRECT SPLIT (KEY FIX)
# =========================
if os.path.exists(SPLIT_PATH):
    print("Loading existing splits...")
    df = pd.read_csv(SPLIT_PATH)
else:
    print("Creating retrieval-friendly splits...")

    df["split"] = "train"

    for item_id in df["item_id"].unique():
        idxs = df[df["item_id"] == item_id].index.tolist()

        if len(idxs) < 2:
            continue

        np.random.shuffle(idxs)

        split_point = int(0.8 * len(idxs))

        gallery_idxs = idxs[:split_point]
        query_idxs = idxs[split_point:]

        df.loc[gallery_idxs, "split"] = "gallery"
        df.loc[query_idxs, "split"] = "query"

    df.to_csv(SPLIT_PATH, index=False)
    print("Splits saved!")

print(df["split"].value_counts())

# IMPORTANT: train uses ALL data (not just train split)
train_df = df.copy().reset_index(drop=True)

# =========================
# DATASET
# =========================
class FashionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.label_map = {k: i for i, k in enumerate(df["item_id"].unique())}
        self.df["label"] = self.df["item_id"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(row["label"])
        }

train_dataset = FashionDataset(train_df)

# =========================
# PK SAMPLER
# =========================
id_to_indices = defaultdict(list)

for idx in range(len(train_dataset)):
    item_id = train_dataset.df.iloc[idx]["item_id"]
    id_to_indices[item_id].append(idx)

class PKSampler(Sampler):
    def __init__(self, id_to_indices, batch_size, k):
        self.id_to_indices = id_to_indices
        self.item_ids = list(id_to_indices.keys())
        self.batch_size = batch_size
        self.k = k

    def __iter__(self):
        batch = []
        random.shuffle(self.item_ids)

        for item_id in self.item_ids:
            indices = self.id_to_indices[item_id]

            if len(indices) < self.k:
                continue

            sampled = random.sample(indices, self.k)
            batch.extend(sampled)

            if len(batch) >= self.batch_size:
                yield from batch[:self.batch_size]
                batch = []

    def __len__(self):
        return len(self.item_ids)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=PKSampler(id_to_indices, BATCH_SIZE, K),
    num_workers=4
)

# =========================
# MODEL
# =========================
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
).to(device)

# Freeze text encoder
for p in model.text_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.vision_model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

# =========================
# LOAD CHECKPOINT
# =========================
start_epoch = 0
best_loss = float("inf")

if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_loss = ckpt["best_loss"]

    print(f"Resuming from epoch {start_epoch}")

# =========================
# LOSS
# =========================
def contrastive_loss(features, labels, temp=0.07):
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T)

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = sim / temp
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)

    mask *= logits_mask
    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    return -mean_log_prob_pos.mean()

# =========================
# TRAIN
# =========================
for epoch in range(start_epoch, EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)

        with torch.cuda.amp.autocast():
            feats = model.vision_model(pixel_values=x).pooler_output
            loss = contrastive_loss(feats, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, CHECKPOINT_PATH)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved best model!")

print("Training complete!")