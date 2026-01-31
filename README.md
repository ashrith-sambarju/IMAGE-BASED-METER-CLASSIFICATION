# AI-Based Meter Classification System

## Project Overview
This project implements an **image-based smart meter classification system** using **Deep Learning (CNNs)**. Given an input image of an electricity meter, the system predicts the **meter brand/type** with a confidence score.

* **Model Performance:** ~94% accuracy on unseen test images.
* **Scale:** Trained and evaluated on 38 different meter classes.
* **Deployment:** Includes a professional Streamlit dashboard for real-time inference.

## Problem Statement
Electricity meters from different manufacturers often have **visually similar designs**, making manual identification slow and error-prone. The goal of this project is to:
* Automatically identify the **meter type** from an image.
* Handle variations such as **blur, zoom, angle, and lighting**.
* Provide **confidence-aware predictions** suitable for real-world deployment.

## Solution Approach

### Model Architecture
* **Backbone:** ResNet-18
* **Framework:** PyTorch
* **Clustering:** HDBSCAN (for unknown images)
* **Input Size:** 224 × 224 RGB images
* **Output:** 38-meter classes via SoftMax
* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam

### Training Strategy
A **two-phase training strategy** was used to improve generalization and reduce overfitting:
1. **Phase 1 – Classifier Training:** ResNet backbone frozen; only the final classification layer trained.
2. **Phase 2 – Fine Tuning:** Last ResNet block unfrozen; end-to-end fine-tuning with a lower learning rate.

## Dataset & Splitting Logic
* **Total Images:** 1,975
* **Total Classes:** 38
* **Strategy:** Because meter data is **highly imbalanced**, we used **adaptive splits**:
  * **Low image count:** Train only.
  * **Medium count:** Train + Test.
  * **Large count:** Train + Val + Test.
* **Audit:** Automated image-wise inference audit on test data replaces manual checking.

## High-Level Logic & "UNKNOWN" Strategy
The system uses a safe fallback mechanism for unseen meters:
1. **Confidence Threshold:** If Top-1 confidence < 50%, the image is marked as **UNKNOWN**.
2. **Metadata Retention:** Even for UNKNOWN images, we store Top-3 predictions and SoftMax confidences. "UNKNOWN" means the model is unsure, not blind.
3. **Clustering:** We extract **embedding vectors** from UNKNOWN images using the CNN backbone and group them using **HDBSCAN**.
   * `cluster_id = -1`: Noise/junk images.
   * `cluster_id >= 0`: New meter candidates for dataset expansion.

##  Project Structure
```text
METER CLASSIFICATION/
├── src/
│   ├── train.py                     # Trains the ResNet18 classifier
│   ├── evaluate.py                  # Generates accuracy and confusion matrix
│   ├── predict.py                   # Single image inference
│   ├── url_predict_batch.py         # Batch URL inference with UNKNOWN logic
│   ├── extract_unknown_embeddings.py# Extracts features for unsure images
│   ├── cluster_unknown_embeddings.py# Runs HDBSCAN clustering
│   └── dashboard.py                 # Streamlit UI
├── data/                            # Processed train/test/val splits
├── models/                          # final_meter_model.pth
└── reports/                         # Metadata, embeddings, and clusters
