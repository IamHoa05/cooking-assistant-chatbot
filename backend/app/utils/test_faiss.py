# test_faiss_improved.py
import pandas as pd
import numpy as np
from faiss_handler import FAISSHandler
from embedder import load_embedding_model, embed_texts
from sklearn.metrics.pairwise import cosine_similarity
import ast

# -----------------------------
# 1️⃣ Load DataFrame và canonicalize ingredients
# -----------------------------
df = pd.read_pickle("./data/recipes_embeddings.pkl")

def canonicalize_ingredient_list(col):
    result = []
    for x in col:
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    val = [v.strip().lower() for v in val]
                else:
                    val = []
            except:
                val = [v.strip().lower() for v in x.split(",") if v.strip()]
        elif isinstance(x, list):
            val = [v.strip().lower() for v in x]
        else:
            val = []
        result.append(val)
    return result

df["ingredient_names"] = canonicalize_ingredient_list(df["ingredient_names"])

# -----------------------------
# 2️⃣ Load FAISS handler
# -----------------------------
faiss_handler = FAISSHandler(df, index_dir="./new_faiss_indexes")

# -----------------------------
# 3️⃣ Nhập nguyên liệu từ user
# -----------------------------
user_input = input("👉 Ingredients (comma separated): ")
user_ingredients = [x.strip().lower() for x in user_input.split(",") if x.strip()]

# -----------------------------
# 4️⃣ Encode input bằng BGE-M3
# -----------------------------
model = load_embedding_model("BAAI/bge-m3")
query_vec = embed_texts(user_ingredients, model)
query_vec = np.mean(query_vec, axis=0)  # trung bình vector của từng nguyên liệu

# -----------------------------
# 5️⃣ Search FAISS index cho ingredient_names
# -----------------------------
top_faiss = 100
top_k = 10
results = faiss_handler.search(query_vec, column_key="names", top_k=top_faiss)

# -----------------------------
# 6️⃣ Tính score cải tiến
# -----------------------------
def compute_scores(recipe_ing, query_ing):
    recipe_set = set(recipe_ing)
    query_set = set(query_ing)
    # Exact match ratio
    exact = len(recipe_set & query_set) / len(query_set) if query_set else 0
    # Jaccard index
    jaccard = len(recipe_set & query_set) / len(recipe_set | query_set) if recipe_set else 0
    return exact, jaccard

for r in results:
    exact, jaccard = compute_scores(r["ingredient_names"], user_ingredients)
    # final_score: cosine * 0.7 + exact * 0.2 + jaccard * 0.1
    r["final_score"] = r["_distance"] * 0.7 + exact * 0.2 + jaccard * 0.1
    r["exact_match"] = exact
    r["jaccard"] = jaccard

# -----------------------------
# 7️⃣ Sắp xếp và lấy top-k món ăn
# -----------------------------
TOP_K = 5

# Lọc ra các món Exact match = 1
exact_match_recipes = [r for r in results if r["exact_match"] == 1.0]

if len(exact_match_recipes) >= TOP_K:
    # Nếu đủ >= TOP_K, lấy top TOP_K theo final_score
    final_recipes = sorted(exact_match_recipes, key=lambda x: x["final_score"], reverse=True)[:TOP_K]
else:
    # Nếu không đủ, lấy tất cả món Exact match = 1
    final_recipes = sorted(exact_match_recipes, key=lambda x: x["final_score"], reverse=True)

print(f"\n🎯 Top gợi ý món ăn (ưu tiên Exact match = 1, max {TOP_K} món):")
for r in final_recipes:
    print(f"- {r['dish_name']}  |  final_score={r['final_score']:.4f}")
    print(f"  Ingredients: {r['ingredient_names']}")
    print(f"  Cosine: {r['_distance']:.4f} | Exact match: {r['exact_match']:.2f} | Jaccard: {r['jaccard']:.2f}\n")