import os
import shutil
import random

# =========================
# PATHS (UPDATED)
# =========================
SOURCE_DIR = "data/raw/meter_data"
TARGET_DIR = "data/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def make_dir(path):
    os.makedirs(path, exist_ok=True)

# Create split folders
for split in ["train", "val", "test"]:
    make_dir(os.path.join(TARGET_DIR, split))

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    n = len(images)
    random.shuffle(images)

    train_dir = os.path.join(TARGET_DIR, "train", cls)
    val_dir   = os.path.join(TARGET_DIR, "val", cls)
    test_dir  = os.path.join(TARGET_DIR, "test", cls)

    make_dir(train_dir)

    # 🔹 Tiny classes → train only
    if n < 10:
        for img in images:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(train_dir, img)
            )
        print(f"{cls}: {n} → train only")
        continue

    # 🔹 Medium classes → train + val
    if n < 20:
        split_train = int(0.8 * n)
        make_dir(val_dir)

        for img in images[:split_train]:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, img))
        for img in images[split_train:]:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, img))

        print(f"{cls}: {n} → train/val")
        continue

    # 🔹 Normal classes → train + val + test
    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    make_dir(val_dir)   
    make_dir(test_dir)

    for img in images[:train_end]:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, img))
    for img in images[train_end:val_end]:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, img))
    for img in images[val_end:]:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, img))

    print(f"{cls}: {n} → train/val/test")

print("\nDataset split completed!")
