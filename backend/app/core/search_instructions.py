# app/services/dish_search_service.py

import os
import yaml
import pandas as pd
import numpy as np
import faiss

from app.utils.embedder import load_embedding_model, embed_texts

# 1. LOAD CONFIG
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yml"))
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBED_MODEL_NAME = config["embedding"]["model_name"]
BATCH_SIZE = config["embedding"].get("batch_size", 32)


# 2. LOAD DATAFRAME
DF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/data/recipes_embeddings.pkl"))
df = pd.read_pickle(DF_PATH)

# Kiểm tra cột embedding
if "dish_name_embedding" not in df.columns:
    raise ValueError("Column 'dish_name_embedding' not found in DataFrame.")

# Lấy danh sách tên món và vectors
dish_names = df["dish_name"].astype(str).tolist()
dish_vectors = np.stack(df["dish_name_embedding"].values).astype("float32")


# 3. NORMALIZE VECTORS (cosine similarity)
# FAISS IndexFlatIP dùng inner product để approximate cosine similarity
dish_vectors = dish_vectors / np.linalg.norm(dish_vectors, axis=1, keepdims=True)


# 4. BUILD FAISS INDEX
dim = dish_vectors.shape[1]
dish_index = faiss.IndexFlatIP(dim)  # IP ~ cosine similarity
dish_index.add(dish_vectors)


# 5. LOAD EMBEDDING MODEL 1 LẦN
embedding_model = load_embedding_model(EMBED_MODEL_NAME)


# 6. HELPER FUNCTION: SEARCH DISH
def search_dish(user_input: str, top_k: int = 1, score_threshold: float = 0.7):
    """
    Tìm món ăn dựa trên input text, trả về danh sách món ăn phù hợp.

    Args:
        user_input (str): câu nhập của người dùng (tên món hoặc nguyên liệu)
        top_k (int): số lượng kết quả tối đa trả về
        score_threshold (float): ngưỡng cosine similarity tối thiểu

    Returns:
        List[dict]: danh sách món ăn, mỗi món chứa:
            - dish_name: tên món
            - score: cosine similarity
            - metadata: dict chứa ingredients, instructions, tips, image_link
    """
    # 1. Encode input thành embedding
    query_vec = embed_texts([user_input], embedding_model, batch_size=BATCH_SIZE, text_type="dish")[0]
    query_vec = np.array(query_vec, dtype="float32").reshape(1, -1)
    query_vec /= np.linalg.norm(query_vec)  # normalize để cosine similarity

    # 2. Search FAISS index
    distances, indices = dish_index.search(query_vec, top_k)

    # 3. Build kết quả trả về, lọc theo score_threshold
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if score < score_threshold:  # lọc kết quả quá thấp
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


# 7. EXAMPLE USAGE
if __name__ == "__main__":
    user_input = "gà chiên, khoai tây"
    results = search_dish(user_input, top_k=5, score_threshold=0.65)

    print(f"Kết quả tìm kiếm cho: {user_input}")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['dish_name']} | Score: {r['score']:.4f}")
        print(f"   Nguyên liệu: {', '.join(r['metadata']['ingredients'])}")
        print(f"   Instructions: {r['metadata']['instructions']}")
        print(f"   Image link: {r['metadata']['image_link']}")
        print()
