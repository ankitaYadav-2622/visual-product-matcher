import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load product data
df = pd.read_csv("product.csv")

embeddings_list = []

print(" Creating embeddings for product images...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row['imagepath']  # NOTE: you changed this column in your CSV

    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
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
        print(f"Could not process {image_path}: {e}")
        embeddings_list.append(np.zeros((512,)))

# Save numpy embeddings
np.save("embeddings.npy", np.array(embeddings_list))
print("embeddings.npy saved successfully")

# Save embeddings alongside product info (optional)
df["embedding"] = [vec.tolist() for vec in embeddings_list]
df.to_json("products_with_embeddings.json", orient="records")
print("products_with_embeddings.json created with embeddings")
