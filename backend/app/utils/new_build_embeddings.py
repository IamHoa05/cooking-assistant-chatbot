import pandas as pd
from new_embedder import load_embedding_model, embed_texts
import numpy as np

# Load CSV
df = pd.read_csv("./data/test_recipes_501_1000_remove_null.csv")

# Load BGE-M3
model = load_embedding_model("BAAI/bge-m3")

# 1. dish_name
dish_vecs = embed_texts(df['dish_name'].fillna("").tolist(), model)
df['dish_name_embedding'] = [vec.tolist() for vec in dish_vecs]

# 2. ingredient_quantities
qty_vecs = embed_texts(df['ingredient_quantities'].fillna("").tolist(), model)
df['ingredient_quantities_embedding'] = [vec.tolist() for vec in qty_vecs]

# 3. ingredient_names
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

# Lưu pickle
df.to_pickle("./new_data/recipes_embeddings.pkl")
print("✅ Saved recipes_embeddings.pkl")
