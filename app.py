import streamlit as st
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="🖼️",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
DATA_CSV = "product.csv"
EMBEDDINGS_FILE = "embeddings.npy"

# read dataset
df = pd.read_csv(DATA_CSV)

# normalize column names
df.columns = df.columns.str.lower().str.strip()
if "imagepath" not in df.columns:
    if "image_path" in df.columns:
        df.rename(columns={"image_path": "imagepath"}, inplace=True)
    elif "image" in df.columns:
        df.rename(columns={"image": "imagepath"}, inplace=True)

# load embeddings
embeddings = np.load(EMBEDDINGS_FILE)

# -------------------------------
# DEVICE + CLIP MODEL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# STYLING
# -------------------------------
st.markdown(
    """
    <style>
    .product-card {
        border-radius: 12px;
        background: #ffffff;
        padding: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .product-card:hover {
        transform: scale(1.05);
    }
    .product-card img {
        max-width: 100%;
        max-height: 200px;
        object-fit: contain;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def search(query, top_k=12):
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        query_features = model.encode_text(text)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_features = query_features.cpu().numpy()

    sims = cosine_similarity(query_features, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_idx].to_dict(orient="records")

# -------------------------------
# APP UI
# -------------------------------
st.title("🔍 Visual Product Matcher")
st.write("Search products by text and find visually similar matches using CLIP.")

query = st.text_input("Enter search query (e.g., 'red shoes')", "")

if query:
    results = search(query, top_k=12)

    cols = st.columns(4, gap="large")

    for idx, item in enumerate(results):
        with cols[idx % 4]:
            with st.container():
                # pick name smartly
                if "product_name" in item:
                    name = item["product_name"]
                elif "category" in item:
                    name = f"{item['category']} #{idx+1}"
                else:
                    name = f"Product #{idx+1}"

                st.markdown(
                    f"""
                    <div class="product-card">
                        <img src="{item.get('imagepath', '')}" alt="{name}" />
                        <p style="text-align:center; margin-top:5px;">{name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
