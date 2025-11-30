# app/services/dish_search_service.py
import pandas as pd
import numpy as np
import faiss
import os
import yaml

from app.utils.embedder import load_embedding_model, embed_texts

# ============================
# LOAD CONFIG
# ============================
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yml"))
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBED_MODEL_NAME = config["embedding"]["model_name"]
BATCH_SIZE = config["embedding"].get("batch_size", 32)

# ============================
# LOAD DATAFRAME
# ============================
DF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/data/recipes_embeddings.pkl"))
df = pd.read_pickle(DF_PATH)

# df cần có các cột:
# - dish_name: str
# - dish_name_embedding: list/np.array (embedding vector)
if "dish_name_embedding" not in df.columns:
    raise ValueError("Column 'dish_name_embedding' not found in DataFrame.")

dish_names = df["dish_name"].astype(str).tolist()
dish_vectors = np.stack(df["dish_name_embedding"].values).astype("float32")

# ============================
# NORMALIZE VECTORS (cosine similarity)
# ============================
dish_vectors = dish_vectors / np.linalg.norm(dish_vectors, axis=1, keepdims=True)

# ============================
# BUILD FAISS INDEX
# ============================
dim = dish_vectors.shape[1]
dish_index = faiss.IndexFlatIP(dim)  # IP = inner product ~ cosine similarity
dish_index.add(dish_vectors)

# ============================
# LOAD EMBEDDING MODEL 1 LẦN
# ============================
embedding_model = load_embedding_model(EMBED_MODEL_NAME)

# ============================
# HELPER FUNCTION
# ============================
def search_dish(user_input: str, top_k: int = 1, score_threshold: float = 0.7):
    query_vec = embed_texts([user_input], embedding_model, batch_size=BATCH_SIZE, text_type="dish")[0]
    query_vec = np.array(query_vec, dtype="float32").reshape(1, -1)
    query_vec /= np.linalg.norm(query_vec)

    distances, indices = dish_index.search(query_vec, top_k)
    results = []

    for idx, score in zip(indices[0], distances[0]):
        if score < score_threshold:
            continue
        row = df.iloc[idx]
        metadata = {
            "ingredients": row.get("ingredient_names", []),
            "instructions": row.get("instructions", []),
            "tips" : row.get("tips", []),
            "image_link" : row.get("image_link", "")
        }
        results.append({
            "dish_name": row["dish_name"],
            "score": float(score),
            "metadata": metadata
        })

    return results