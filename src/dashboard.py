import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(         
    page_title="Smart Meter Classification",
    page_icon="⚡",
    layout="wide"
)

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/final_meter_model.pth"
DATA_DIR = "data/processed"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚡ Meter Classifier")
st.sidebar.markdown(
    f"""
    **Deep Learning based meter classification system**

    - Model: ResNet18  
    - Classes: 38  
    - Test Accuracy: **94%**  
    - Device: `{DEVICE.upper()}`  
    """
)

st.sidebar.markdown("---")
# st.sidebar.markdown("Built by **Ashrith**")

# =========================
# LOAD CLASSES
# =========================
class_names = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
num_classes = len(class_names)  

# =========================
# TRANSFORMS
# =========================
infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =========================
# HELPERS
# =========================
def load_image_any(src):
    """Load image from URL or local path safely"""
    try:
        if str(src).startswith("http"):
            r = requests.get(src, timeout=5)
            return Image.open(BytesIO(r.content)).convert("RGB")
        else:
            return Image.open(src).convert("RGB")
    except Exception:
        return None

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📷 Single Image Prediction", "📊 Batch Results Viewer"])

# =========================================================
# TAB 1 — SINGLE IMAGE
# =========================================================
with tab1:
    st.markdown("<h2 style='text-align:center;'>🔌 Single Meter Image Classification</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Meter Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", width=300)

        img_tensor = infer_tfms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        conf, pred_idx = torch.max(probs, dim=0)
        pred_class = class_names[pred_idx.item()]

        with col2:
            st.markdown("### Prediction")
            st.success(pred_class)

            st.markdown("### Confidence")
            st.progress(float(conf))
            st.write(f"**{conf.item()*100:.2f}%**")

        # Top-3
        st.markdown("### 🏆 Top-3 Predictions")
        top3 = torch.topk(probs, 3)

        rows = []
        for i in range(3):
            rows.append({
                "Rank": i + 1,
                "Class": class_names[top3.indices[i].item()],
                "Confidence (%)": round(top3.values[i].item() * 100, 2)
            })

        st.table(pd.DataFrame(rows))

        if conf.item() < 0.7:
            st.warning("⚠️ Low confidence prediction — image may be blurred or ambiguous.")
    else:
        st.info("Upload a meter image to start prediction")

# =========================================================
# TAB 2 — BATCH RESULTS VIEWER
# =========================================================
with tab2:
    st.markdown("<h2 style='text-align:center;'>📊 Batch Inference Audit</h2>", unsafe_allow_html=True)
    st.caption("Upload batch inference CSV (URL → prediction results)")

    csv_file = st.file_uploader(
        "📥 Upload Batch Results CSV",
        type=["csv"]
    )

    if csv_file:
        df = pd.read_csv(csv_file)

        required_cols = {
            "image_url", "final_prediction",
            "top1_class", "top1_confidence",
            "top2_class", "top2_confidence",
            "top3_class", "top3_confidence"
        }

        if not required_cols.issubset(df.columns):
            st.error("❌ CSV format invalid. Required columns missing.")
            st.stop()

        batch_classes = sorted(df["final_prediction"].unique())
        selected_class = st.selectbox("Filter by class", batch_classes)

        min_conf = st.slider(
            "Minimum confidence (%)",
            0, 100, 0, 5
        )

        filtered = df[
            (df["final_prediction"] == selected_class) &
            (df["top1_confidence"] >= min_conf)
        ]

        st.markdown(f"### Showing `{len(filtered)}` images for **{selected_class}**")

        cols_per_row = 4
        for i in range(0, len(filtered), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, (_, r) in zip(cols, filtered.iloc[i:i+cols_per_row].iterrows()):
                with col:
                    img = load_image_any(r["image_url"])
                    if img:
                        st.image(img, width=250)
                    else:
                        st.error("Image not found")

                    st.markdown(
                        f"""
                        **Final:** `{r['final_prediction']}`  
                        **Top-1:** `{r['top1_class']}` ({r['top1_confidence']}%)
                        """
                    )

                    with st.expander("Top-3"):
                        st.write(
                            f"""
                            1️⃣ {r['top1_class']} — {r['top1_confidence']}%  
                            2️⃣ {r['top2_class']} — {r['top2_confidence']}%  
                            3️⃣ {r['top3_class']} — {r['top3_confidence']}%
                            """
                        )

        st.download_button(
            "⬇️ Download Filtered Results",
            filtered.to_csv(index=False),
            file_name=f"{selected_class}_results.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload batch inference CSV to visualize results")
