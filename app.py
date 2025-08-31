import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import os

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="🖼️",
    layout="wide"
)

# -------------------------------
# SIDEBAR: THEME SWITCH
# -------------------------------
st.sidebar.header("⚙️ Display Options")
theme_choice = st.sidebar.radio("Choose Theme:", ["☀️ Light", "🌙 Dark"])

if theme_choice == "☀️ Light":
    style_block = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        color: black;
    }
    [data-testid="stHeader"] { background: transparent; }
    h1 { color: #2E86C1; font-weight: bold; }
    .product-card {
        background: white;
        border-radius: 15px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .product-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    </style>
    """
else:  # 🌙 Dark
    style_block = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #1e1e1e 0%, #2c3e50 100%);
        color: white;
    }
    [data-testid="stHeader"] { background: transparent; }
    h1 { color: #F39C12; font-weight: bold; }
    .product-card {
        background: #2c3e50;
        border-radius: 15px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .product-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 18px rgba(0,0,0,0.8);
    }
    </style>
    """

st.markdown(style_block, unsafe_allow_html=True)

# -------------------------------
# LOAD DATA & CLIP MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return clip.load("ViT-B/32", device)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_model()

# ✅ Try loading dataset, else fallback to demo data
if os.path.exists("product.csv") and os.path.exists("embeddings.npy"):
    products_df = pd.read_csv("product.csv")
    emb_matrix = np.load("embeddings.npy")
else:
    st.warning("⚠️ product.csv or embeddings.npy not found. Using demo sample data.")
    products_df = pd.DataFrame({
        "product_name": ["Demo Product 1", "Demo Product 2"],
        "imagepath": [
            "https://via.placeholder.com/150",
            "https://via.placeholder.com/150"
        ]
    })
    emb_matrix = np.random.rand(len(products_df), 512)

# -------------------------------
# FEATURE EXTRACTION HELPERS
# -------------------------------
def get_features_from_file(file_obj):
    try:
        img = Image.open(file_obj).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = clip_model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().flatten()
    except Exception as e:
        st.error(f" Failed to process image: {e}")
        return None

def get_features_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = clip_model.encode_image(tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().flatten(), img
    except Exception as e:
        return None, None

# -------------------------------
# UI: TITLE + TABS
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🖼️ Visual Product Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Discover products that look alike in just a click 🚀</p>", unsafe_allow_html=True)
st.markdown("---")

tab_upload, tab_url = st.tabs(["📂 Upload Image", "🌐 Use Image URL"])

query_features = None

# --- Upload Tab ---
with tab_upload:
    file_input = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if file_input:
        st.image(file_input, caption="Uploaded Image", width=300)
        query_features = get_features_from_file(file_input)

# --- URL Tab ---
with tab_url:
    url_input = st.text_input("Paste an image URL")
    if url_input:
        vec, url_img = get_features_from_url(url_input)
        if vec is not None:
            st.image(url_img, caption="Image from URL", width=300)
            query_features = vec
        else:
            st.error(" Could not fetch or process image from the URL.")

# -------------------------------
# SIMILARITY SEARCH
# -------------------------------
if query_features is not None:
    with st.spinner("🔍 Searching for similar products..."):
        sim_scores = cosine_similarity([query_features], emb_matrix)[0]
        top_n = 50
        ranked_idx = np.argsort(sim_scores)[::-1][:top_n]

    threshold = st.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Only show results above this similarity"
    )

    st.subheader("Top Matching Products")
    grid_cols = st.columns(4)
    shown_count = 0

    for i, idx in enumerate(ranked_idx):
        if sim_scores[idx] < threshold:
            continue

        item = products_df.iloc[idx]
        col = grid_cols[shown_count % 4]

        with col:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)

            image_path = item['imagepath']
            try:
                st.image(image_path, width=150)
            except Exception:
                st.warning(f"⚠️ Image not found: {image_path}")

            st.markdown(f"""
            <h4>{item['product_name']}</h4>
            Similarity: {sim_scores[idx]:.3f}
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        shown_count += 1

    if shown_count == 0:
        st.warning("⚠️ No items matched the selected similarity threshold.")
