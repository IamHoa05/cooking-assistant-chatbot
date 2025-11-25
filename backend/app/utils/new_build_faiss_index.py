# new_build_faiss_index.py
import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from new_embedder import load_embedding_model, embed_texts


INDEX_DIR = "./new_faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


def build_faiss_index(vectors: np.ndarray, output_path: str):
    """
    Build chỉ số FAISS với cosine similarity (IP + normalized).
    """
    dim = vectors.shape[1]

    print(f"🔧 Building FAISS index: {output_path}")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, output_path)
    print(f"✅ Saved FAISS index → {output_path}")


def build_all_indexes(df: pd.DataFrame):

    model = load_embedding_model("BAAI/bge-m3")

    # --------------------------
    # 1. dish_name
    # --------------------------
    print("\n🔹 Embedding dish_name ...")
    dish_vecs = embed_texts(df["dish_name"].fillna("").tolist(), model)
    np.save(f"{INDEX_DIR}/dish_name_embedding.npy", dish_vecs)
    build_faiss_index(dish_vecs, f"{INDEX_DIR}/dish_name_embedding.index")

    # --------------------------
    # 2. ingredient_quantities
    # --------------------------
    print("\n🔹 Embedding ingredient_quantities ...")
    qty_vecs = embed_texts(df["ingredient_quantities"].fillna("").tolist(), model)
    np.save(f"{INDEX_DIR}/ingredient_quantities_embedding.npy", qty_vecs)
    build_faiss_index(qty_vecs, f"{INDEX_DIR}/ingredient_quantities_embedding.index")

    # --------------------------
    # 3. ingredient_names (flatten)
    # --------------------------
    print("\n🔹 Embedding ingredient_names (flattening)...")

    flat_vecs = []
    flat_row_ids = []  # map về row gốc

    for row_idx, ing_list in tqdm(df["ingredient_names"].items()):
        if isinstance(ing_list, str):
            ing_list = [x.strip() for x in ing_list.split(",") if x.strip()]
        elif not isinstance(ing_list, list):
            ing_list = []

        if len(ing_list) > 0:
            vecs = embed_texts(ing_list, model)
            flat_vecs.extend(vecs)
            flat_row_ids.extend([row_idx] * len(vecs))

    flat_vecs = np.array(flat_vecs, dtype="float32")
    flat_row_ids = np.array(flat_row_ids)

    np.save(f"{INDEX_DIR}/ingredient_names_embedding.npy", flat_vecs)
    np.save(f"{INDEX_DIR}/ingredient_names_row_ids.npy", flat_row_ids)

    build_faiss_index(flat_vecs, f"{INDEX_DIR}/ingredient_names_embedding.index")

    print("\n🎉 DONE BUILDING ALL INDEXES")


if __name__ == "__main__":
    df = pd.read_csv("./data/test_recipes_501_1000_remove_null.csv")
    build_all_indexes(df)
