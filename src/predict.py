import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image       
import argparse

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/final_meter_model.pth"
DATA_DIR = "data/processed"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD CLASS NAMES (FROM TRAIN)
# =========================
train_classes = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
num_classes = len(train_classes)

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
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = infer_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    return train_classes[pred_idx.item()], conf.item()

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to meter image")
    args = parser.parse_args()

    pred, conf = predict_image(args.image)
    print(f"\nPrediction: {pred}")
    print(f"Confidence: {conf*100:.2f}%")
