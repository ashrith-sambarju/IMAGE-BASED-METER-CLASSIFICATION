import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np

# =========================
# CONFIG          
# =========================
DATA_DIR = "data/processed"    
BATCH_SIZE = 8   
IMG_SIZE = 224

EPOCHS_HEAD = 6        # phase 1: classifier only
EPOCHS_FINETUNE = 8    # phase 2: unfreeze layer4
    
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models", exist_ok=True)

print("Using device:", DEVICE)
  
# =========================
# TRANSFORMS
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), 
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                 
        std=[0.229, 0.224, 0.225]
    )
])
           
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS
# =========================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tfms)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)   

val_loader = DataLoader(     
    val_ds,
    batch_size=BATCH_SIZE,  
    shuffle=False,
    num_workers=0
)  

num_classes = len(train_ds.classes)
print("Number of classes:", num_classes)

# =========================
# CLASS WEIGHTS (SMOOTHED)
# =========================
labels = [label for _, label in train_ds.samples]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights = class_weights ** 0.75   #smoothing
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# =========================
# MODEL   
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ---- Freeze entire backbone ----
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Unfreeze classifier only
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# =========================
# PHASE 1: TRAIN CLASSIFIER
# =========================
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD)

print("\n🔹 Phase 1: Training classifier head only")

for epoch in range(EPOCHS_HEAD):
    model.train()
    correct, total, train_loss = 0, 0, 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_HEAD}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.4f}")

# =========================
# PHASE 2: FINE-TUNE LAYER4
# =========================
print("\n🔹 Phase 2: Fine-tuning last ResNet block")

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FINETUNE
)

for epoch in range(EPOCHS_FINETUNE):
    model.train()
    correct, total, train_loss = 0, 0, 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Finetune {epoch+1}/{EPOCHS_FINETUNE}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.4f}")

# =========================
# SAVE FINAL MODEL
# =========================
torch.save(model.state_dict(), "models/final_meter_model.pth")
print("\n Final model saved: models/final_meter_model.pth")
