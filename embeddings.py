import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# -------------------------------
# CONFIG
# -------------------------------
INPUT_CSV = "product.csv"        # your big file (already exists)
OUTPUT_CSV = "product.csv"       # overwrite with 180 sampled items
OUTPUT_NPY = "embeddings.npy"    # embeddings for 180 items
OUTPUT_JSON = "products_with_embeddings.json"

N_SAMPLES = 180                  # how many items you want

# -------------------------------
# DEVICE + CLIP MODEL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# LOAD + SAMPLE DATA
# -------------------------------
df = pd.read_csv(INPUT_CSV)

# normalize column names
df.columns = df.columns.str.lower().str.strip()
if "imagepath" not in df.columns:
    if "image_path" in df.columns:
        df.rename(columns={"image_path": "imagepath"}, inplace=True)
    elif "image" in df.columns:
        df.rename(columns={"image": "imagepath"}, inplace=True)

# randomly pick 180 items
df_small = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

# -------------------------------
# CREATE EMBEDDINGS
# -------------------------------
embeddings_list = []
print(f"🔍 Creating embeddings for {len(df_small)} images...")

for idx, row in tqdm(df_small.iterrows(), total=len(df_small)):
    image_path = row["imagepath"]

    if not os.path.isfile(image_path):
        print(f"⚠️ File not found: {image_path}")
        embeddings_list.append(np.zeros((512,)))
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        tensor_img = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            vector = model.encode_image(tensor_img)
            vector = vector / vector.norm(dim=-1, keepdim=True)
            embeddings_list.append(vector.cpu().numpy()[0])
    except Exception as e:
        print(f"❌ Could not process {image_path}: {e}")
        embeddings_list.append(np.zeros((512,)))

# -------------------------------
# SAVE RESULTS
# -------------------------------
np.save(OUTPUT_NPY, np.array(embeddings_list))
print(f"✅ Saved embeddings to {OUTPUT_NPY}")

df_small.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Overwrote {OUTPUT_CSV} with reduced dataset ({N_SAMPLES} rows)")

df_small["embedding"] = [vec.tolist() for vec in embeddings_list]
df_small.to_json(OUTPUT_JSON, orient="records")
print(f"✅ Saved JSON with embeddings to {OUTPUT_JSON}")
