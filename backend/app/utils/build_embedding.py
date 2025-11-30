import os
import numpy as np
import pandas as pd

from embedder import load_embedding_model, embed_texts


# ============================================================
# 1. Load file CSV
# ============================================================
df_path = os.path.join("./data/recipes_cleaned.csv")
df = pd.read_csv(df_path)


# ============================================================
# 2. Load mô hình BGE-M3 từ config.yml
#    (model_name và batch_size đều lấy trong config)
# ============================================================
model = load_embedding_model()


# ============================================================
# 3. Tính embedding cho tên món ăn (dish_name)
# ============================================================
dish_names = df["dish_name"].fillna("").tolist()
dish_vecs = embed_texts(dish_names, model)

# Convert thành list để lưu vào pickle
df["dish_name_embedding"] = [vec.tolist() for vec in dish_vecs]


# ============================================================
# 4. Tính embedding cho ingredient_names (danh sách nguyên liệu)
# ============================================================
ingredient_embeddings_list = []

for ing_list in df["ingredient_names"]:

    # Chuẩn hóa thành list
    if isinstance(ing_list, str):
        ing_list = [x.strip() for x in ing_list.split(",") if x.strip()]
    elif not isinstance(ing_list, list):
        ing_list = []

    # Tính embedding từng nguyên liệu
    if ing_list:
        vecs = embed_texts(ing_list, model)
        vecs = [v.tolist() for v in vecs]  # convert sang list
    else:
        vecs = []

    ingredient_embeddings_list.append(vecs)

df["ingredient_names_embedding"] = pd.Series(
    ingredient_embeddings_list, dtype=object
)


# ============================================================
# 5. Lưu kết quả vào pickle
# ============================================================
output_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "recipes_embeddings.pkl")
df.to_pickle(output_path)

print("✅ Saved recipes_embeddings.pkl")
