import pandas as pd
from embedder import load_embedding_model, embed_texts
import numpy as np
import os

# ===== Load CSV =====
df_path = os.path.join("./data/recipes_cleaned.csv")
df = pd.read_csv(df_path)

# ===== Load BGE-M3 từ config =====
model = load_embedding_model()  # model_name và batch_size lấy từ config.yml

# ===== 1. dish_name =====
dish_vecs = embed_texts(df['dish_name'].fillna("").tolist(), model)
df['dish_name_embedding'] = [vec.tolist() for vec in dish_vecs]

# ===== 2. ingredient_names =====
ingredient_embeddings_list = []

for ing_list in df['ingredient_names']:
    if isinstance(ing_list, str):
        ing_list = [x.strip() for x in ing_list.split(",") if x.strip()]
    elif not isinstance(ing_list, list):
        ing_list = []

    if ing_list:
        vecs = embed_texts(ing_list, model)  
        vecs = [v.tolist() for v in vecs]
    else:
        vecs = []

    ingredient_embeddings_list.append(vecs)

df['ingredient_names_embedding'] = pd.Series(ingredient_embeddings_list, dtype=object)

# ===== Lưu pickle =====
output_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(output_dir, exist_ok=True)
df.to_pickle(os.path.join(output_dir, "recipes_embeddings.pkl"))

print("✅ Saved recipes_embeddings.pkl")
