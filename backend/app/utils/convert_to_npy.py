# test_faiss_with_pickle.py
import pandas as pd
import numpy as np
from new_faiss_handler import FAISSHandler
from new_embedder import load_embedding_model, embed_texts
import ast

# ---------------------------
# 1️⃣ Load embeddings từ pickle
# ---------------------------
pickle_path = "./data/recipes_embeddings.pkl"
df = pd.read_pickle(pickle_path)

# Nếu ingredient_names trong pickle vẫn là string, parse sang list
def parse_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []

df['ingredient_names'] = df['ingredient_names'].apply(parse_list)

print(f"Loaded {len(df)} recipes from pickle")

# ---------------------------
# 2️⃣ Load embedding model
# ---------------------------
model = load_embedding_model("BAAI/bge-m3")

# ---------------------------
# 3️⃣ Load FAISS indexes
# ---------------------------
handler = FAISSHandler(df, index_dir="./new_faiss_indexes")

# ---------------------------
# 4️⃣ Input user
# ---------------------------
ingredients_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
ingredients = [x.strip() for x in ingredients_input.split(",") if x.strip()]

# ---------------------------
# 5️⃣ Encode input
# ---------------------------
query_vecs = embed_texts(ingredients, model)

# ---------------------------
# 6️⃣ Search FAISS (ingredient_names)
# ---------------------------
# Flatten vectors + row map, lấy trung bình
results_dict = {}
for vec in query_vecs:
    results = handler.search(vec, column_key="names", top_k=20)
    for r in results:
        rowid = r["_rowid"]
        if rowid not in results_dict:
            results_dict[rowid] = {"recipe": r, "count": 0}
        results_dict[rowid]["count"] += 1

# ---------------------------
# 7️⃣ Tính final score (cosine + match ratio)
# ---------------------------
final_results = []
for rowid, info in results_dict.items():
    recipe = info["recipe"]
    match_ratio = info["count"] / len(ingredients)
    cosine = recipe["_distance"]
    final_score = 0.7 * cosine + 0.3 * match_ratio
    final_results.append({
        "dish_name": recipe["dish_name"],
        "ingredients": recipe["ingredient_names"],
        "cosine": cosine,
        "match_ratio": match_ratio,
        "final_score": final_score
    })

# Sắp xếp theo final_score
final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)

# ---------------------------
# 8️⃣ In ra top 5
# ---------------------------
print("\n🎯 Top gợi ý món ăn:")
for r in final_results[:5]:
    print(f"- {r['dish_name']}  |  final_score={r['final_score']:.4f}")
    print(f"  Ingredients: {r['ingredients']}")
    print(f"  Cosine: {r['cosine']:.4f} | Match ratio: {r['match_ratio']:.2f}\n")
