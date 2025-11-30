import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
INDEX_DIR = "./faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


# ============================================================
# HÀM: Build FAISS index đơn (cosine similarity)
# ============================================================
def build_faiss_index(vectors: np.ndarray, output_path: str):
    """
    Xây dựng FAISS index dùng Cosine Similarity.
    Thực tế: FAISS không có cosine trực tiếp nên ta dùng:
        cosine = inner_product(normalized_vectors)
    """

    dim = vectors.shape[1]
    print(f"🔧 Building FAISS index: {output_path}")

    # FAISS Index với Inner Product
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Lưu index
    faiss.write_index(index, output_path)
    print(f"✅ Saved FAISS index → {output_path}")


# ============================================================
# HÀM: Build toàn bộ index từ file pickle
# ============================================================
def build_all_indexes_from_pkl(pkl_path: str):
    """Load embeddings từ file PKL và build tất cả FAISS indexes."""

    print(f"📂 Loading embeddings from {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # --------------------------------------------------------
    # 1. FAISS index cho dish_name_embedding
    # --------------------------------------------------------
    print("\n🔹 Building FAISS index for dish_name ...")

    # df['dish_name_embedding'] là list[list_float]
    dish_vecs = np.vstack(df["dish_name_embedding"].values).astype("float32")

    # Lưu NPY cho debug / tái sử dụng
    np.save(f"{INDEX_DIR}/dish_name_embedding.npy", dish_vecs)

    # Build FAISS index
    build_faiss_index(
        dish_vecs,
        f"{INDEX_DIR}/dish_name_embedding.index"
    )

    # --------------------------------------------------------
    # 2. FAISS index cho ingredient_names_embedding (flatten)
    # --------------------------------------------------------
    print("\n🔹 Building FAISS index for ingredient_names (flattening) ...")

    flat_vecs = []
    flat_row_ids = []

    # flatten vì mỗi món có nhiều nguyên liệu → nhiều vector
    for row_idx, vec_list in tqdm(df["ingredient_names_embedding"].items()):

        if isinstance(vec_list, list) and len(vec_list) > 0:
            for v in vec_list:
                flat_vecs.append(v)          # vector nguyên liệu → thêm vào danh sách chung
                flat_row_ids.append(row_idx) # lưu id món tương ứng

    # Convert sang numpy
    flat_vecs = np.array(flat_vecs, dtype="float32")
    flat_row_ids = np.array(flat_row_ids)

    # Lưu raw vectors + row mapping
    np.save(f"{INDEX_DIR}/ingredient_names_embedding.npy", flat_vecs)
    np.save(f"{INDEX_DIR}/ingredient_names_row_ids.npy", flat_row_ids)

    # Build FAISS index
    build_faiss_index(
        flat_vecs,
        f"{INDEX_DIR}/ingredient_names_embedding.index"
    )

    print("\n🎉 DONE BUILDING ALL INDEXES FROM PKL")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pkl_path = "./data/recipes_embeddings.pkl"
    build_all_indexes_from_pkl(pkl_path)
