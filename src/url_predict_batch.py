import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import requests
from tqdm import tqdm
import os
import cv2
import numpy as np

# =====================
# CONFIG
# =====================
INPUT_FILE = "data/batchtest/28thjan_29thjan_2026_nbpdcl_.csv"   # .csv OR .xlsx
IMAGE_COL = 0                                                   # column index or name
OUTPUT_FILE = "reports/batchtest_predictions_nbpdcl.csv"
MODEL_PATH = "models/final_meter_model.pth"

IMG_SIZE = 224
CONF_THRESHOLD = 0.50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dark / poor-quality thresholds (tuned for your dataset)
MIN_BRIGHTNESS = 38
MIN_CONTRAST = 18
GREEN_RATIO_THRESHOLD = 1.6

print("Using device:", DEVICE)

# =====================
# TRANSFORMS
# =====================
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# LOAD MODEL
# =====================
class_names = sorted(os.listdir("data/processed/train"))
num_classes = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================
# IMAGE LOADER
# =====================
def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

# =====================
# ROBUST FILE LOADER
# =====================
def load_batch_file(path, image_col):
    path_lower = path.lower()

    if path_lower.endswith(".csv"):
        df = pd.read_csv(path, usecols=[image_col])

    elif path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        df = pd.read_excel(
            path,
            usecols=[image_col],
            engine="openpyxl"
        )
    else:
        raise ValueError(
            "Unsupported file format. Only .csv and .xlsx are supported."
        )

    df.columns = ["image_url"]
    return df

# =====================
# DATA QUALITY CHECK
# =====================
def check_data_quality(pil_img):
    img = np.array(pil_img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()
    contrast = gray.std()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()

    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()
    green_ratio = mean_g / (mean_r + mean_b + 1e-6)

    is_dark = (
        brightness < 45 or
        contrast < 22 or
        edge_density < 6 or
        (green_ratio > 1.4 and edge_density < 8)
    )

    return is_dark, {
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "edge_density": round(float(edge_density), 2),
        "green_ratio": round(float(green_ratio), 2),
    }

# =====================
# LOAD INPUT FILE
# =====================
df = load_batch_file(INPUT_FILE, IMAGE_COL)

print(f"Total images found: {len(df)}")

# =====================
# RUN INFERENCE
# =====================
results = []

for url in tqdm(df["image_url"], desc="Running batch inference"):
    try:
        pil_img = load_image_from_url(url)

        # -----------------
        # DATA QUALITY GATE
        # -----------------
        is_dark, metrics = check_data_quality(pil_img)

        if is_dark:
            results.append({
                "image_url": url,
                "final_prediction": "DARK",
                "top1_class": None,
                "top1_confidence": None,
                "top2_class": None,
                "top2_confidence": None,
                "top3_class": None,
                "top3_confidence": None,
                "brightness": metrics["brightness"],
                "contrast": metrics["contrast"],
                "green_ratio": metrics["green_ratio"],
                "reason": "low illumination / low contrast / green glare"
            })
            continue

        # -----------------
        # MODEL INFERENCE
        # -----------------
        img = tfms(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)[0]

        top3_conf, top3_idx = torch.topk(probs, 3)

        top1_conf = top3_conf[0].item()
        top1_class = class_names[top3_idx[0].item()]

        final_label = (
            top1_class if top1_conf >= CONF_THRESHOLD else "UNKNOWN"
        )

        results.append({
            "image_url": url,
            "final_prediction": final_label,
            "top1_class": top1_class,
            "top1_confidence": round(top1_conf * 100, 2),
            "top2_class": class_names[top3_idx[1].item()],
            "top2_confidence": round(top3_conf[1].item() * 100, 2),
            "top3_class": class_names[top3_idx[2].item()],
            "top3_confidence": round(top3_conf[2].item() * 100, 2),
            "brightness": metrics["brightness"],
            "contrast": metrics["contrast"],
            "green_ratio": metrics["green_ratio"],
            "reason": None
        })

    except Exception as e:
        results.append({
            "image_url": url,
            "final_prediction": "ERROR",
            "top1_class": None,
            "top1_confidence": None,
            "top2_class": None,
            "top2_confidence": None,
            "top3_class": None,
            "top3_confidence": None,
            "brightness": None,
            "contrast": None,
            "green_ratio": None,
            "reason": str(e)
        })

# =====================
# SAVE RESULTS
# =====================
os.makedirs("reports", exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

print(f"\nBatch predictions saved to: {OUTPUT_FILE}")