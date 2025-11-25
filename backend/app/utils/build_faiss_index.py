
# build_faiss_indexes.py
import os
import numpy as np
import faiss
import pandas as pd

def build_faiss_indexes(df, index_dir="./faiss_indexes"):
    os.makedirs(index_dir, exist_ok=True)

    # ----- 1. ingredient_names_embedding (flatten từng nguyên liệu) -----
    if "ingredient_names_embedding" in df.columns:
        print("📌 Building FAISS index for ingredient_names_embedding (flattened)")
        flat_embeddings = []
        row_indices = []

        for idx, ing_list in enumerate(df['ingredient_names_embedding']):
            for vec in ing_list:
                flat_embeddings.append(np.array(vec).astype("float32"))
                row_indices.append(idx)  # lưu idx món gốc

        flat_embeddings = np.array(flat_embeddings)
        row_indices = np.array(row_indices)

        dim = flat_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(flat_embeddings)

        faiss.write_index(index, os.path.join(index_dir, "ingredient_names_embedding.index"))
        np.save(os.path.join(index_dir, "ingredient_names_embedding_row_indices.npy"), row_indices)
        print("✅ Saved ingredient_names_embedding index + row indices\n")

    # ----- 2. ingredient_quantities_embedding và dish_name_embedding -----
    for col in ["ingredient_quantities_embedding", "dish_name_embedding"]:
        if col not in df.columns:
            continue
        print(f"📌 Building FAISS index for {col}")
        embeddings = np.array(df[col].tolist()).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(index_dir, f"{col}.index"))
        print(f"✅ Saved {col} index\n")


if __name__ == "__main__":
    df = pd.read_pickle("./data/recipes_embeddings.pkl")
    build_faiss_indexes(df, index_dir="./faiss_indexes")
    print("🎉 All FAISS indexes built successfully!")
