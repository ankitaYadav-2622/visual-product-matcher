# 🖼️ VISUAL IMAGE PRODUCT MATCH

A **Streamlit-based image similarity search tool** that uses **OpenAI CLIP embeddings** and **FAISS** for fast image retrieval.  
You can **add**, **search**, and **reset** an index of images using image or text queries.

---

## ✨ Features

- ✅ Upload and **add images** to a FAISS index  
- 🔎 **Search similar images** by uploading an image or entering a text prompt  
- ♻️ **Reset** the index to remove all stored images  
- 📦 Automatically stores image embeddings using **CLIP (ViT-B/32)**  
- 📁 Supports **batch embedding generation** for datasets  
- 🗂️ Creates **structured CSV files** for image metadata (ID, category, path, etc.)

---

## 🏗️ Project Structure
<pre> project/ │ ├── app/ # Main Streamlit app │ ├── main.py # Streamlit UI │ └── ... │ ├── core/ # Core utilities │ ├── clip_utils.py # CLIP embedding functions │ ├── faiss_manager.py # FAISS index handling │ ├── storage.py # Image save/delete helpers │ └── config.py # Configuration (TOP_K, paths, etc.) │ ├── images/ # Uploaded and indexed images │ ├── generate_embeddings.py # Batch embedding generator ├── generate_csv.py # CSV generator for categorized images │ ├── embeddings.npy # Saved embeddings (generated) ├── product.csv # Image metadata (name, path) ├── products.csv # Image metadata with ID, category, URL │ └── README.md # (This file) </pre>


PROJECT STRUCTURE


## ⚙️ Setup Instructions

### 1. Clone the Repository
git clone https://github.com/yourusername/image-similarity-tool.git
cd image-similarity-tool

### 2. Install Requirements

Use pip to install the dependencies: pip install -r requirements.txt
Make sure you have PyTorch and Streamlit installed:
pip install torch torchvision
pip install streamlit faiss-cpu pillow numpy pandas

### 3. Run the Streamlit App
streamlit run app/main.py

## TECH STACK
Python
Streamlit
OpenAI CLIP (ViT-B/32)
PyTorch
Pillow
NumPy
Pandas
CSV (Python Standard Library)






