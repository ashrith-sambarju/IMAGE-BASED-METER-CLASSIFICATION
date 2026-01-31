import os
import torch
import torch.nn as nn
import pandas as pd
import requests
from io import BytesIO
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_CSV = "reports/batchtest_predictions_tpcodl.csv"   # CSV from url_predict_batch.py
IMAGE_COL = "image_url"
FILTER_CLASS = "UNKNOWN"

MODEL_PATH = "models/final_meter_model.pth"

EMB_OUT_DIR = "reports/embeddings_tpcodl"
META_OUT_DIR = "reports/metadata_tpcodl"       

EMB_OUT_FILE = f"{EMB_OUT_DIR}/unknown_embeddings.csv"
META_OUT_FILE = f"{META_OUT_DIR}/unknown_metadata.csv"

IMG_SIZE = 224    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PREP
# =========================
os.makedirs(EMB_OUT_DIR, exist_ok=True)
os.makedirs(META_OUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(INPUT_CSV)

unknown_df = df[df["final_prediction"] == FILTER_CLASS].reset_index(drop=True)

print(f"Found {len(unknown_df)} UNKNOWN images")

if len(unknown_df) == 0:
    print("No UNKNOWN images found. Exiting.")
    exit()

# =========================
# TRANSFORMS
# =========================
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL (EMBEDDING MODE)
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Identity()  #  remove classifier

state = torch.load(MODEL_PATH, map_location=DEVICE)
state = {k.replace("fc.", ""): v for k, v in state.items() if "fc" not in k}
model.load_state_dict(state, strict=False)

model = model.to(DEVICE)
model.eval()

# =========================
# EXTRACTION
# =========================
emb_rows = []
meta_rows = []

for idx, row in tqdm(unknown_df.iterrows(), total=len(unknown_df)):

    img_url = row[IMAGE_COL]

    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = tfms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy()[0]

        # ---- Embedding row ----
        emb_rows.append({
            "image_url": img_url,
            **{f"f{i}": emb[i] for i in range(len(emb))}
        })

        # ---- Metadata row ----
        meta_rows.append({
            "image_url": img_url,
            "top1_class": row["top1_class"],
            "top1_confidence": row["top1_confidence"],
            "top2_class": row["top2_class"],
            "top2_confidence": row["top2_confidence"],
            "top3_class": row["top3_class"],
            "top3_confidence": row["top3_confidence"],
        })

    except Exception as e:
        print(f" Failed: {img_url} | {e}")

# =========================
# SAVE RESULTS
# =========================
pd.DataFrame(emb_rows).to_csv(EMB_OUT_FILE, index=False)
pd.DataFrame(meta_rows).to_csv(META_OUT_FILE, index=False)

print("\n UNKNOWN embedding extraction completed")
print(f"Embeddings saved to: {EMB_OUT_FILE}")
print(f" Metadata saved to:   {META_OUT_FILE}")
