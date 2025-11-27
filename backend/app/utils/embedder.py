

# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import pandas as pd
# import torch
# from typing import List


# def load_vietnamese_embedding_model(device: str = None):
#     """
#     Load BGE embedding model with GPU support.
#     """
#     model_name = "BAAI/bge-base-en-v1.5"   # hoặc bản fine-tuned tiếng Việt nếu có
#     print(f"🔹 Loading BGE model: {model_name}")

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = SentenceTransformer(model_name, device=device)
#     print(f"👉 Using device: {device}")

#     return model


# def embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 32, text_type: str = "passage") -> List[List[float]]:
#     """
#     Encode danh sách text bằng BGE + normalize embeddings.
#     text_type: 'passage' cho dữ liệu gốc, 'query' cho input user.
#     """
#     prefix = f"{text_type}: "
#     prefixed_texts = [prefix + t for t in texts]

#     embeddings = []
#     for i in tqdm(range(0, len(prefixed_texts), batch_size)):
#         batch = prefixed_texts[i:i + batch_size]
#         batch_embeddings = model.encode(
#             batch,
#             show_progress_bar=False,
#             normalize_embeddings=True
#         )
#         embeddings.extend(batch_embeddings)

#     return [emb.tolist() for emb in embeddings]


# def generate_vietnamese_recipe_embeddings(df: pd.DataFrame):
#     """
#     Tạo embeddings cho từng cột:
#     - ingredient_names: embed từng nguyên liệu riêng lẻ
#     - ingredient_quantities: embed cả cột nếu cần
#     - dish_name: embed tên món
#     """
#     model = load_vietnamese_embedding_model()

#     # 1. Embed dish_name
#     print("Embedding column: dish_name")
#     df['dish_name_embedding'] = embed_texts(
#         df['dish_name'].fillna("").astype(str).tolist(),
#         model,
#         batch_size=32,
#         text_type="passage"
#     )

#     # 2. Embed ingredient_names
#     print("Embedding column: ingredient_names (từng nguyên liệu)")
#     ingredient_embeddings_list = []

#     for ing_list in tqdm(df['ingredient_names'], desc="Processing ingredient_names"):
#         if isinstance(ing_list, str):
#             # Nếu nguyên liệu chưa convert thành list, giả sử tách bởi dấu phẩy
#             ing_list = [x.strip() for x in ing_list.split(",") if x.strip()]
#         elif not isinstance(ing_list, list):
#             ing_list = []

#         # Embed từng nguyên liệu riêng lẻ
#         if ing_list:
#             vecs = embed_texts(ing_list, model, batch_size=16, text_type="passage")
#         else:
#             vecs = []

#         ingredient_embeddings_list.append(vecs)

#     df['ingredient_names_embedding'] = pd.Series(ingredient_embeddings_list, dtype=object)

#     # 3. Optionally embed ingredient_quantities nếu muốn
#     print("Embedding column: ingredient_quantities")
#     df['ingredient_quantities_embedding'] = embed_texts(
#         df['ingredient_quantities'].fillna("").astype(str).tolist(),
#         model,
#         batch_size=32,
#         text_type="passage"
#     )

#     return df


# if __name__ == "__main__":
#     # Load CSV
#     df = pd.read_csv("./data/test_recipes_501_1000_remove_null.csv")

#     # Tạo embeddings
#     df_emb = generate_vietnamese_recipe_embeddings(df)

#     # Lưu DataFrame
#     output = "./data/recipes_embeddings.pkl"
#     df_emb.to_pickle(output)
#     print(f"✅ Saved embeddings to {output}")








# app/utils/embedder.py

import os
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


# -------------------------
# 1. Load model nhẹ < 100MB
# -------------------------
def load_vietnamese_embedding_model(device: str = None):
    """
    Model embed nhẹ cho tiếng Việt (~40MB)
    Chạy nhanh – độ chính xác cao – không tốn RAM
    """
    model_name = "intfloat/multilingual-e5-small"   # < 50MB, support tiếng Việt rất tốt
    print(f"🔹 Loading embedding model: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["HF_HOME"] = "D:/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"

    model = SentenceTransformer(model_name, device=device)
    print(f"👉 Using device: {device}")

    return model


# -------------------------
# 2. Embed text + normalize
# -------------------------
def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    mode: str = "passage"
):
    """
    Encode text dạng E5:
    - query: "query: ... "
    - passage: "passage: ... "
    """
    prefix = "query: " if mode == "query" else "passage: "
    texts = [prefix + t for t in texts]

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(emb.tolist())

    return vectors


# -------------------------
# 3. Generate embeddings
# -------------------------
def generate_vietnamese_recipe_embeddings(df: pd.DataFrame):
    model = load_vietnamese_embedding_model()

    # ---- dish_name ----
    print("📌 Embedding dish_name…")
    df["dish_name_embedding"] = embed_texts(
        df["dish_name"].fillna("").astype(str).tolist(),
        model,
        mode="passage"
    )

    # ---- ingredient_names: embed từng NGUYÊN LIỆU RIÊNG ----
    print("📌 Embedding ingredient_names…")
    ing_embeds = []

    for ing_list in tqdm(df["ingredient_names"]):
        if isinstance(ing_list, str):
            ing_list = [x.strip() for x in ing_list.split(",") if x.strip()]
        if not isinstance(ing_list, list):
            ing_list = []

        # Embed từng nguyên liệu
        ing_embeds.append(embed_texts(ing_list, model, batch_size=32, mode="passage"))

    df["ingredient_names_embedding"] = ing_embeds

    # ---- ingredient_quantities ----
    print("📌 Embedding ingredient_quantities…")
    df["ingredient_quantities_embedding"] = embed_texts(
        df["ingredient_quantities"].fillna("").astype(str).tolist(),
        model,
        mode="passage"
    )

    return df
