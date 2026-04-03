# 👗 Visual Fashion Product Retrieval using CLIP

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Transformers-CLIP-yellow?style=for-the-badge&logo=huggingface"/>
  <img src="https://img.shields.io/badge/Task-Image%20Retrieval-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-Accelerated-green?style=for-the-badge"/>
</p>

---

## 📌 Overview

This project implements a **visual product retrieval system** for fashion images using **CLIP (Contrastive Language–Image Pretraining)**.

Given a query image, the system retrieves visually similar products from a gallery using learned embeddings.

---

## ✨ Features

- 🔍 Image-to-image retrieval using deep embeddings  
- 🧠 Fine-tuned CLIP vision encoder  
- ⚡ Contrastive learning with PK sampling  
- 📊 Evaluation using Recall@K, mAP@K, NDCG@K  
- 🖼️ Visual retrieval results with similarity scores  
- 🚀 GPU-accelerated training and inference  

---

## 🗂️ Dataset

- **Dataset Used:** DeepFashion  
- Total images: ~52,000+  
- Each product has multiple views (front, side, back, etc.)

### 📌 Split Strategy (IMPORTANT)

For each product (`item_id`):

- 80% images → **Gallery**  
- 20% images → **Query**  

✔ Ensures same product appears in both sets (required for retrieval)

---

## 🧠 Methodology

### 🔹 Model

- Pretrained: `openai/clip-vit-base-patch32`  
- Fine-tuned: Vision encoder only  

---

### 🔹 Training Strategy

- Contrastive Learning  
- PK Sampling (multiple samples per product per batch)  
- Normalized embeddings  
- Cosine similarity for retrieval  

---

### 🔹 Loss Function

Contrastive loss encourages:

- Similar images → closer embeddings  
- Different products → farther apart  

---

## 📊 Evaluation Metrics

| Metric | @5 | @10 | @15 |
|--------|----|-----|-----|
| **Recall** | 0.947 | 0.967 | 0.976 |
| **mAP** | 0.846 | 0.810 | 0.787 |
| **NDCG** | 0.882 | 0.875 | 0.869 |

---

## 🖼️ Sample Results

<p align="center">
  <img src="assets/Figure_1.png" width="800"/>
</p>
<p align="center">
  <img src="assets/Figure_2.png" width="800"/>
</p>
<p align="center">
  <img src="assets/Figure_3.png" width="800"/>
</p>
<p align="center">
  <img src="assets/Figure_4.png" width="800"/>
</p>
<p align="center">
  <img src="assets/Figure_5.png" width="800"/>
</p>


- 🔵 Query image on left  
- 🟢 Green border = correct match  
- 🔴 Red border = incorrect match  
- 🔢 Similarity scores shown  

---

## NOTES:
Please download the best model and the last epoch's checkpoint from this sharepoint here(only visible to iiitb email accounts):
https://iiitbac-my.sharepoint.com/:f:/g/personal/yashub_goel_iiitb_ac_in/IgBn1tcLyvBHSJ7mWOy1NdldAWhwERnLzkR6m9X_Ehk-F-k?e=cfRlBB
