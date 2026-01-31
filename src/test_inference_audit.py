import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd

# =========================
# CONFIG
# =========================
TEST_DIR = "data/processed/test"
TRAIN_DIR = "data/processed/train"
MODEL_PATH = "models/final_meter_model.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD CLASS NAMES (FROM TRAIN!)
# =========================
class_names = sorted(os.listdir(TRAIN_DIR))
num_classes = len(class_names)

print("Using device:", DEVICE)
print("Model classes:", num_classes)

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
# LOAD MODEL (38 classes)
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model = model.to(DEVICE)
model.eval()

# =========================
# INFERENCE AUDIT
# =========================
records = []
total = 0
correct = 0

test_classes = sorted(os.listdir(TEST_DIR))

for true_class in test_classes:
    class_dir = os.path.join(TEST_DIR, true_class)

    if not os.path.isdir(class_dir):
        continue

    for img_name in tqdm(os.listdir(class_dir), desc=f"Testing {true_class}"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = infer_tfms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        pred_class = class_names[pred_idx.item()]
        is_correct = pred_class == true_class

        records.append({
            "image": img_name,
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": round(conf.item(), 4),
            "correct": is_correct
        })

        total += 1
        correct += int(is_correct)

# =========================
# RESULTS
# =========================
df = pd.DataFrame(records)

accuracy = correct / total
print(f"\nOverall Test Accuracy (image-wise): {accuracy:.4f}")

print("\nPer-class accuracy:")
print(
    df.groupby("true_class")["correct"]
      .mean()
      .sort_values(ascending=False)
)

# =========================
# SAVE RESULTS
# =========================
os.makedirs("reports", exist_ok=True)
df.to_csv("reports/test_inference_results.csv", index=False)

print("\nDetailed results saved to:")
print("reports/test_inference_results.csv")
