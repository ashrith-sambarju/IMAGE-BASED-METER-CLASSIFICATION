
# Smart Meter Image Classification using Deep Learning

## Project Overview

This project implements an **image-based smart meter classification system** using **Deep Learning (CNNs)**.
Given an input image of an electricity meter, the system predicts the **meter brand/type** with a confidence score.

The model is trained and evaluated on **38 different meter classes**, achieving **~94.7% accuracy on unseen test images**.
A **professional Streamlit dashboard** is also provided for real-time inference.

---

## Problem Statement

Electricity meters from different manufacturers often have **visually similar designs**, making manual identification slow and error-prone.

The goal of this project is to:

* Automatically identify the **meter type** from an image
* Handle variations such as **blur, zoom, angle, and lighting**
* Provide **confidence-aware predictions** suitable for real-world deployment

---

## Solution Approach

### Why Image Classification (Not Object Detection)?

* Meter images are **already well-cropped**
* No need for bounding boxes or localization
* Classification is **simpler, faster, and more robust**
* Achieves high accuracy without annotation overhead

---

## Model Architecture

* **Backbone:** ResNet-18
* **Framework:** PyTorch
* **Input Size:** 224 × 224 RGB images
* **Output:** 38 meter classes
* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam

---

## Training Strategy

A **two-phase training strategy** was used:

### Phase 1 – Classifier Training

* ResNet backbone frozen
* Only the final classification layer trained

### Phase 2 – Fine Tuning

* Last ResNet block unfrozen
* End-to-end fine tuning with lower learning rate

This approach significantly improved generalization and reduced overfitting.

---

## Dataset Details

* **Total Images:** 1,975
* **Total Classes:** 38
* **Train / Validation / Test Split**

  * Classes with very few samples were used for **training only**
  * **24 classes** were used for final test evaluation

Directory structure:

```
data/processed/
├── train/
├── val/
└── test/
```

Folder names represent **ground truth labels**.

---

## Evaluation Results

### Overall Performance

* **Test Accuracy (image-wise):** **94.7%**

### Per-Class Highlights

* Many classes achieved **100% accuracy**
* Lower performance observed only in:

  * Visually similar meters
  * Classes with fewer samples

### Error Analysis

* Errors are **low-confidence predictions**
* No systematic failure patterns
* Model confidence correlates well with correctness

---

## Additional Validation

Beyond standard evaluation:

* Manual random testing using unseen images
* Automated **image-wise inference audit** on test data
* Detailed CSV report generated for inspection

---

## Dashboard (Real-Time Inference)

A **Streamlit-based dashboard** allows:

* Uploading a new meter image
* Viewing predicted class and confidence
* Inspecting top-3 predictions
* Flagging low-confidence predictions

Run the dashboard:

```bash
python -m streamlit run src/dashboard.py
```

---

## Project Structure

```
METER CLASSIFICATION/
│
├── data/
│   └── processed/
│
├── models/
│   └── final_meter_model.pth
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dashboard.py
│   └── test_inference_audit.py
│
├── reports/
│   └── test_inference_results.csv
│
├── requirements.txt
└── README.md
```
---

## Installation & Setup

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\Activate.ps1  # Windows
```

### Install Dependencies (CUDA)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate.py
```

### Predict Single Image

```bash
python src/predict.py --image path/to/image.jpg
```

### Run Inference Audit

```bash
python src/test_inference_audit.py
```

### Launch Dashboard

```bash
python -m streamlit run src/dashboard.py
```

---

## Key Learnings

* Transfer learning significantly improves performance with limited data
* Confidence-aware predictions are critical for real-world systems
* Proper dataset splitting avoids misleading evaluation
* Clean environment management is essential for reproducibility

---

## Future Enhancements

* Add data augmentation for hard classes
* Temperature scaling for confidence calibration
* Mobile / web deployment
* Continuous learning with new data

---
