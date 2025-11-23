# import os
# import faiss
# import numpy as np
# import pandas as pd


# # =============================
# #  VALIDATION FUNCTIONS
# # =============================

# def clean_embeddings_list(raw_vectors):
#     """
#     Validate + convert embedding list → numpy array
#     Trả về:
#         embeddings: np.ndarray float32 (N, dim)
#         valid_indexes: list index hợp lệ
#     """

#     cleaned = []
#     valid_indexes = []

#     for i, v in enumerate(raw_vectors):
#         if v is None:
#             print(f"⚠️ Embedding row {i} is None → skipped")
#             continue

#         arr = np.asarray(v)

#         if arr.ndim != 1:
#             print(f"⚠️ Embedding row {i} shape invalid: {arr.shape} → skipped")
#             continue

#         cleaned.append(arr)
#         valid_indexes.append(i)

#     if len(cleaned) == 0:
#         raise ValueError("❌ Không có embedding hợp lệ nào để build FAISS index!")

#     # Convert to matrix
#     embeddings = np.stack(cleaned).astype("float32")
#     print(f"🔧 Cleaned embeddings shape: {embeddings.shape}")

#     return embeddings, valid_indexes


# # =============================
# #  BUILD FAISS INDEX
# # =============================

# def build_faiss_index(embeddings: np.ndarray,
#                       output_path: str,
#                       use_ivf: bool = False,
#                       nlist: int = 100):
#     """
#     Build FAISS index:
#         - Flat (mặc định)
#         - IVF (nhanh hơn khi dataset lớn)
#     """

#     dim = embeddings.shape[1]

#     if use_ivf:
#         print("⚡ Using IVF index for faster search")

#         quantizer = faiss.IndexFlatL2(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist)

#         print("🔹 Training IVF index...")
#         index.train(embeddings)

#     else:
#         print("🔹 Using exact FAISS IndexFlatL2")
#         index = faiss.IndexFlatL2(dim)

#     print("📌 Adding vectors to FAISS index...")
#     index.add(embeddings)

#     print(f"💾 Saving FAISS index → {output_path}")
#     faiss.write_index(index, output_path)

#     print(f"✅ Done. Total vectors stored: {index.ntotal}\n")


# # =============================
# #  MAIN PROCESS: BUILD 3 INDEXES
# # =============================

# def build_all_indexes(df_path="./data/recipes_embeddings.pkl",
#                       output_dir="./data",
#                       use_ivf=False):
#     """
#     Build FAISS index cho 3 embedding:
#         - dish_name_embedding
#         - ingredient_names_embedding
#         - ingredient_quantities_embedding
#     """

#     print(f"📥 Loading dataframe: {df_path}")
#     df = pd.read_pickle(df_path)

#     os.makedirs(output_dir, exist_ok=True)

#     embedding_columns = {
#         "dish_name_embedding": "faiss_index_dish_name.bin",
#         "ingredient_names_embedding": "faiss_index_ingredients.bin",
#         "ingredient_quantities_embedding": "faiss_index_quantities.bin",
#     }

#     for col, file_name in embedding_columns.items():

#         print(f"\n==========================")
#         print(f"🔹 Building FAISS for column: {col}")
#         print("==========================")

#         if col not in df.columns:
#             print(f"❌ Column missing → skipped: {col}")
#             continue

#         raw_vectors = df[col].tolist()

#         # Validate + clean embedding
#         embeddings, valid_ids = clean_embeddings_list(raw_vectors)

#         # Save FAISS index
#         output_path = os.path.join(output_dir, file_name)
#         build_faiss_index(
#             embeddings,
#             output_path,
#             use_ivf=use_ivf
#         )


# if __name__ == "__main__":
#     build_all_indexes(use_ivf=False)  # đổi True nếu muốn IVF speedup


# import os
# import faiss
# import numpy as np
# import pandas as pd

# # =============================
# #  VALIDATION FUNCTIONS
# # =============================
# def clean_embeddings_list(raw_vectors):
#     cleaned = []
#     valid_indexes = []

#     for i, v in enumerate(raw_vectors):
#         if v is None:
#             continue
#         arr = np.asarray(v, dtype=np.float32)
#         if arr.ndim != 1:
#             continue
#         # --- normalize để dùng cosine ---
#         # arr /= np.linalg.norm(arr)
#         cleaned.append(arr)
#         valid_indexes.append(i)

#     if len(cleaned) == 0:
#         raise ValueError("❌ Không có embedding hợp lệ nào để build FAISS index!")

#     embeddings = np.stack(cleaned)
#     print(f"🔧 Cleaned embeddings shape: {embeddings.shape}")
#     return embeddings, valid_indexes

# =============================
#  BUILD FAISS INDEX (COSINE)
# =============================
# def build_faiss_index(embeddings: np.ndarray,
#                       output_path: str,
#                       use_ivf: bool = False,
#                       nlist: int = 100):
#     dim = embeddings.shape[1]

#     if use_ivf:
#         print("⚡ Using IVF index (cosine) for faster search")
#         quantizer = faiss.IndexFlatIP(dim)  # IP = cosine after normalization
#         index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#         print("🔹 Training IVF index...")
#         index.train(embeddings)
#     else:
#         print("🔹 Using exact FAISS IndexFlatIP (cosine)")
#         index = faiss.IndexFlatIP(dim)

#     print("📌 Adding vectors to FAISS index...")
#     index.add(embeddings)

#     print(f"💾 Saving FAISS index → {output_path}")
#     faiss.write_index(index, output_path)

#     print(f"✅ Done. Total vectors stored: {index.ntotal}\n")


# =============================
#  MAIN PROCESS: BUILD ALL INDEXES
# =============================
# def build_all_indexes(df_path="./data/recipes_embeddings.pkl",
#                       output_dir="./data",
#                       use_ivf=False):
#     print(f"📥 Loading dataframe: {df_path}")
#     df = pd.read_pickle(df_path)

#     os.makedirs(output_dir, exist_ok=True)

#     embedding_columns = {
#         "dish_name_embedding": "faiss_index_dish_name.bin",
#         "ingredient_names_embedding": "faiss_index_ingredients.bin",
#         "ingredient_quantities_embedding": "faiss_index_quantities.bin",
#     }

#     for col, file_name in embedding_columns.items():
#         print(f"\n==========================")
#         print(f"🔹 Building FAISS for column: {col}")
#         print("==========================")

#         if col not in df.columns:
#             print(f"❌ Column missing → skipped: {col}")
#             continue

#         raw_vectors = df[col].tolist()
#         embeddings, valid_ids = clean_embeddings_list(raw_vectors)

#         output_path = os.path.join(output_dir, file_name)
#         build_faiss_index(
#             embeddings,
#             output_path,
#             use_ivf=use_ivf
#         )


# if __name__ == "__main__":
#     build_all_indexes(use_ivf=False)

import os
import faiss
import numpy as np
import pandas as pd
import math

# =============================
#  VALIDATION + NORMALIZATION
# =============================
# def clean_embeddings_list(raw_vectors):
#     cleaned = []
#     valid_indexes = []

#     for i, v in enumerate(raw_vectors):
#         if v is None:
#             continue
#         arr = np.asarray(v, dtype=np.float32)
#         if arr.ndim != 1:
#             continue

#         # --- Normalize để dùng cosine ---
#         norm = np.linalg.norm(arr)
#         if norm == 0:
#             continue
#         arr = arr / norm

#         cleaned.append(arr)
#         valid_indexes.append(i)

#     if not cleaned:
#         raise ValueError("❌ Không có embedding hợp lệ!")

#     embeddings = np.stack(cleaned)
#     print(f"🔧 Cleaned embeddings shape: {embeddings.shape}")
#     return embeddings, valid_indexes


# =============================
#  BUILD FAISS INDEX (BEST WAY)
# =============================
# def build_faiss_index_best(embeddings: np.ndarray,
#                            output_path: str,
#                            force_ivf: bool = None):
#     N, dim = embeddings.shape
#     print(f"📌 Building FAISS (N={N:,}, d={dim})")

#     # -----------------------------
#     # Auto-select index type
#     # -----------------------------
#     if force_ivf is not None:
#         use_ivf = force_ivf
#     else:
#         use_ivf = N >= 200_000  # auto choose

#     # IVF parameters (best practice)
#     nlist = int(math.sqrt(N))
#     nlist = max(64, min(nlist, 4096))  # clamp tốt nhất

#     if use_ivf:
#         print(f"⚡ Using IVF index (nlist={nlist})")
#         # quantizer = faiss.IndexFlatIP(dim)
#         # index = faiss.IndexIVFFlat(
#         #     quantizer,
#         #     dim,
#         #     nlist,
#         #     faiss.METRIC_INNER_PRODUCT,
#         # )
#         # L2 distance
#         quantizer = faiss.IndexFlatL2(dim)
#         index = faiss.IndexIVFFlat(
#             quantizer,
#             dim,
#             nlist,
#             faiss.METRIC_L2,
#         )

#         print("🔹 Training IVF...")
#         index.train(embeddings)

#         # Tăng accuracy (best practice)
#         index.nprobe = int(math.sqrt(nlist))
#         print(f"🔧 nprobe set to {index.nprobe}")

#     else:
#         print("🎯 Using IndexFlatIP (exact, 100% accuracy)")
#         index = faiss.IndexFlatIP(dim)

#     # -----------------------------
#     print("📥 Adding vectors...")
#     index.add(embeddings)

#     # -----------------------------
#     print(f"💾 Saving → {output_path}")
#     faiss.write_index(index, output_path)
#     print(f"✅ Done. Total vectors = {index.ntotal}\n")

def build_faiss_indexes(df, config):
    """
    Tạo FAISS index cho nhiều cột embedding trong DataFrame.

    Các cột embedding cần khớp chính xác với file đã embed:
        - ingredient_names_embedding
        - ingredient_quantities_embedding
        - dish_name_embedding
    """

    # Map cột DataFrame → tên file index
    columns_to_embed = {
        "ingredient_names_embedding": "ingredient_names_embedding",
        "ingredient_quantities_embedding": "ingredient_quantities_embedding",
        "dish_name_embedding": "dish_name_embedding",
    }

    index_dir = config["paths"]["faiss_index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    for col_name, file_name in columns_to_embed.items():

        print(f"📌 Building FAISS index for column: {col_name}")

        if col_name not in df.columns:
            raise KeyError(f"❌ Cột '{col_name}' không tồn tại trong DataFrame!")

        # Convert list → np.array float32
        embeddings = np.array(df[col_name].tolist()).astype("float32")

        dim = embeddings.shape[1]

        # Dùng IndexFlatIP vì bạn đang normalize → tương đương cosine
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Lưu file .index
        index_path = os.path.join(index_dir, f"{file_name}.index")
        faiss.write_index(index, index_path)

        print(f"✅ Saved: {index_path}\n")


if __name__ == "__main__":
    df = pd.read_pickle("./data/recipes_embeddings.pkl")

    config = {
        "paths": {
            "faiss_index_dir": "./faiss_indexes"
        }
    }

    build_faiss_indexes(df, config)

# # =============================
# #  MAIN: BUILD ALL INDEXES
# # =============================
# def build_all_indexes(df_path="./data/recipes_embeddings.pkl",
#                       output_dir="./data",
#                       force_ivf=None):
#     print(f"📥 Loading dataframe: {df_path}")
#     df = pd.read_pickle(df_path)

#     os.makedirs(output_dir, exist_ok=True)

#     embedding_columns = {
#         "dish_name_embedding": "faiss_index_dish_name.bin",
#         "ingredient_names_embedding": "faiss_index_ingredients.bin",
#         "ingredient_quantities_embedding": "faiss_index_quantities.bin",
#     }

#     for col, file_name in embedding_columns.items():
#         print("\n==========================")
#         print(f"🔹 Building FAISS for column: {col}")
#         print("==========================")

#         if col not in df.columns:
#             print(f"❌ Missing → skipped: {col}")
#             continue

#         raw_vectors = df[col].tolist()
#         embeddings, valid_ids = clean_embeddings_list(raw_vectors)

#         output_path = os.path.join(output_dir, file_name)
#         build_faiss_index_best(
#             embeddings,
#             output_path,
#             force_ivf=force_ivf
#         )


# # =============================
# if __name__ == "__main__":
#     # Auto-select IVF nếu dataset lớn
#     build_all_indexes(force_ivf=None)



