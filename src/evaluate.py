import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =========================
# CONFIG
# =========================
DATA_DIR = "data/processed"
MODEL_PATH = "models/final_meter_model.pth"
BATCH_SIZE = 8
IMG_SIZE = 224
TRAIN_CLASSES = 38   # MUST match training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================
test_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_tfms
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_classes = test_ds.classes
print("Test classes:", len(test_classes))

# =========================
# CLASS ALIGNMENT
# =========================
train_classes = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
class_to_train_idx = {cls: i for i, cls in enumerate(train_classes)}
test_to_train_idx = {i: class_to_train_idx[cls] for i, cls in enumerate(test_classes)}

# =========================
# MODEL (MATCH TRAINING)
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, TRAIN_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================
# EVALUATION
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(1)

        mapped_labels = [test_to_train_idx[int(l)] for l in labels.cpu().numpy()]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(mapped_labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================
# METRICS
# =========================
accuracy = (all_preds == all_labels).mean()
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=list(test_to_train_idx.values()),
    target_names=test_classes,
    digits=4
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(test_to_train_idx.values())
)

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    xticklabels=test_classes,
    yticklabels=test_classes,
    cmap="Blues",
    annot=False
)
plt.title("Confusion Matrix - Meter Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# TOP CONFUSED PAIRS
# =========================
print("\n Top Confused Class Pairs:")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

pairs = []
for i in range(len(test_classes)):
    for j in range(len(test_classes)):
        if cm_no_diag[i, j] > 0:
            pairs.append((test_classes[i], test_classes[j], cm_no_diag[i, j]))

pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]

for a, b, c in pairs:
    print(f"{a} → {b} : {c}")
