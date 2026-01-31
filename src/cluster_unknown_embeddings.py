# src/cluster_unknown_embeddings.py

import os
import hashlib
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_EMB_CSV = "reports/embeddings_tpcodl/unknown_embeddings.csv"
OUTPUT_DIR = "reports/clustered_tpcodl"
CLUSTER_IMG_DIR = f"{OUTPUT_DIR}/clusters"

MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 4
IMG_TIMEOUT = 10  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLUSTER_IMG_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("Loading embeddings...")
df = pd.read_csv(INPUT_EMB_CSV)

feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].values

print(f"Total UNKNOWN samples: {len(X)}")

# =========================
# SCALE EMBEDDINGS
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# HDBSCAN CLUSTERING
# =========================
print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean"
)

df["cluster_id"] = clusterer.fit_predict(X_scaled)

# =========================
# SAVE CLUSTERED CSVs
# =========================
clustered_csv = f"{OUTPUT_DIR}/unknown_clustered.csv"
df.to_csv(clustered_csv, index=False)   
print(f"Clustered results saved to: {clustered_csv}")

summary = (
    df.groupby("cluster_id")
      .size()
      .reset_index(name="count")
      .sort_values("cluster_id")
)

summary_csv = f"{OUTPUT_DIR}/cluster_summary.csv"
summary.to_csv(summary_csv, index=False)

print("\nCluster Summary:")
print(summary)

# =========================
# DOWNLOAD IMAGES INTO CLUSTERS
# =========================
print("\nDownloading images into cluster folders...")

for cid in sorted(df["cluster_id"].unique()):
    if cid == -1:
        continue  # skip noise

    cluster_path = f"{CLUSTER_IMG_DIR}/cluster_{cid}"
    os.makedirs(cluster_path, exist_ok=True)

clustered_rows = df[df["cluster_id"] >= 0]

for _, row in tqdm(clustered_rows.iterrows(), total=len(clustered_rows)):
    url = row["image_url"]
    cid = row["cluster_id"]

    # Safe unique filename
    fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    out_path = f"{CLUSTER_IMG_DIR}/cluster_{cid}/{fname}"

    if os.path.exists(out_path):
        continue

    try:
        r = requests.get(url, timeout=IMG_TIMEOUT)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(r.content)

    except Exception as e:
        print(f"Failed: {url} | {e}")

print("\nDone ✅")
print("\nNOTES:")
print("- cluster_id = -1  → true OUTLIERS / bad images")
print("- cluster_id >= 0 → visually inspect & label as NEW meter types")
        