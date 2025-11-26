# import pandas as pd
# import numpy as np

# # ===== ĐƯỜNG DẪN FILE EMBEDDING (.pkl) =====
# pkl_path = "./data/recipes_embeddings.pkl"

# # ===== LOAD DỮ LIỆU EMBEDDINGS =====
# df_emb = pd.read_pickle(pkl_path)
# print("✅ Loaded embeddings dataframe:", df_emb.shape)

# # ===== CHỌN CÁC CỘT EMBEDDING CẦN DÙNG =====
# embedding_cols = [
#     "ingredient_names_embedding",
#     "ingredient_quantities_embedding",
#     "dish_name_embedding"
# ]

# # ===== GỘP (CONCATENATE) TẤT CẢ EMBEDDING CỘT LẠI THÀNH 1 VECTOR DUY NHẤT =====
# # Mỗi embedding là 1 list hoặc ndarray → chuyển sang numpy rồi nối lại
# combined_embeddings = np.concatenate(
#     [np.vstack(df_emb[col].values) for col in embedding_cols], axis=1
# )

# # ===== CHUYỂN SANG DẠNG float32 (FAISS YÊU CẦU) =====
# embedding_matrix = combined_embeddings.astype("float32")

# # ===== LƯU RA FILE .NPY =====
# output_path = "./data/recipes_embeddings.npy"
# np.save(output_path, embedding_matrix)

# print(f"✅ Saved numpy embeddings to {output_path}")
# print(f"📏 Shape: {embedding_matrix.shape}")

import pandas as pd
import numpy as np

# ===== ĐƯỜNG DẪN FILE EMBEDDING (.pkl) =====
pkl_path = "./data/recipes_embeddings.pkl"

# ===== LOAD DỮ LIỆU EMBEDDINGS =====
df_emb = pd.read_pickle(pkl_path)
print("✅ Loaded embeddings dataframe:", df_emb.shape)

# ===== CỘT EMBEDDING CẦN DÙNG =====
embedding_cols = [
    "ingredient_names_embedding",
    "ingredient_quantities_embedding",
    "dish_name_embedding"
]

# ===== CHUẨN HÓA SHAPE MỖI ROW TRONG CỘT =====
all_embeddings = []
for col in embedding_cols:
    shapes = [np.array(row, dtype=np.float32).size for row in df_emb[col].values]
    max_len = max(shapes)
    
    col_emb = []
    for row in df_emb[col].values:
        arr = np.array(row, dtype=np.float32).flatten()
        if arr.size < max_len:
            arr = np.pad(arr, (0, max_len - arr.size))
        col_emb.append(arr)
    col_emb = np.stack(col_emb)
    all_embeddings.append(col_emb)
    
# ===== NỐI TẤT CẢ EMBEDDING LẠI =====
combined_embeddings = np.concatenate(all_embeddings, axis=1)  # shape = (num_recipes, sum_dims)

# ===== LƯU RA FILE .NPY =====
output_path = "./data/recipes_embeddings.npy"
np.save(output_path, combined_embeddings)

print(f"✅ Saved numpy embeddings to {output_path}")
print(f"📏 Shape: {combined_embeddings.shape}")
