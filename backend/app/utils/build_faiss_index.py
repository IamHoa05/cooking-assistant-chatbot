# build_faiss_index.py
import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

INDEX_DIR = "./faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

def build_faiss_index(vectors: np.ndarray, output_path: str):
    """
    Build FAISS index với cosine similarity (IP + normalized vectors).
    """
    dim = vectors.shape[1]
    print(f"🔧 Building FAISS index: {output_path}")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, output_path)
    print(f"✅ Saved FAISS index → {output_path}")


def build_all_indexes_from_pkl(pkl_path: str):
    """
    Build FAISS indexes từ file pickle đã chứa embeddings.
    """
    print(f"📂 Loading embeddings from {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # --------------------------
    # 1. dish_name
    # --------------------------
    print("\n🔹 Building FAISS index for dish_name ...")
    dish_vecs = np.vstack(df['dish_name_embedding'].values).astype("float32")
    np.save(f"{INDEX_DIR}/dish_name_embedding.npy", dish_vecs)
    build_faiss_index(dish_vecs, f"{INDEX_DIR}/dish_name_embedding.index")


    # --------------------------
    # 2. ingredient_names (flatten)
    # --------------------------
    print("\n🔹 Building FAISS index for ingredient_names (flattening)...")

    flat_vecs = []
    flat_row_ids = []

    for row_idx, vec_list in tqdm(df['ingredient_names_embedding'].items()):
        if isinstance(vec_list, list) and len(vec_list) > 0:
            for v in vec_list:
                flat_vecs.append(v)
                flat_row_ids.append(row_idx)

    flat_vecs = np.array(flat_vecs, dtype="float32")
    flat_row_ids = np.array(flat_row_ids)

    np.save(f"{INDEX_DIR}/ingredient_names_embedding.npy", flat_vecs)
    np.save(f"{INDEX_DIR}/ingredient_names_row_ids.npy", flat_row_ids)

    build_faiss_index(flat_vecs, f"{INDEX_DIR}/ingredient_names_embedding.index")

    print("\n🎉 DONE BUILDING ALL INDEXES FROM PKL")


if __name__ == "__main__":
    pkl_path = "./data/recipes_embeddings.pkl"
    build_all_indexes_from_pkl(pkl_path)
