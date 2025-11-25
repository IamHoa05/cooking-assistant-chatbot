import pandas as pd
import numpy as np
from new_embedder import load_embedding_model, embed_texts
from new_faiss_handler import FAISSHandler

# -----------------------------
# Config
# -----------------------------
TOP_K = 10          # số lượng trả về từ FAISS
FINAL_TOP_K = 5     # số lượng in ra
ALPHA = 0.7         # weight cosine similarity
BETA = 0.3          # weight match ratio

# -----------------------------
# Canonical mapping (tùy chỉnh)
# -----------------------------
CANONICAL = {
    "trứng gà": "trứng gà",
    "trứng": "trứng gà",
    "gà": "thịt gà",
    "tôm sú": "tôm",
    "tôm": "tôm",
    "ốc hương": "ốc hương",
    "bí xanh": "bí xanh",
    # thêm các mapping khác nếu cần
}

def to_canonical(name):
    name = name.lower().strip()
    return CANONICAL.get(name, name)

# -----------------------------
# Tokenize + canonical
# -----------------------------
def tokenize_canonical(text):
    tokens = text.lower().split()
    return [to_canonical(t) for t in tokens]

# -----------------------------
# Load dữ liệu và model
# -----------------------------
df = pd.read_csv("./data/test_recipes_501_1000_remove_null.csv")
handler = FAISSHandler(df, index_dir="./new_faiss_indexes")
model = load_embedding_model("BAAI/bge-m3")

# -----------------------------
# Input từ user
# -----------------------------
ingredients_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
query_ingredients = [x.strip() for x in ingredients_input.split(",") if x.strip()]
query_tokens = []
for q in query_ingredients:
    query_tokens.extend(tokenize_canonical(q))

if not query_tokens:
    print("❌ Không có nguyên liệu nào được nhập.")
    exit()

# -----------------------------
# Encode query ingredients (từng nguyên liệu riêng)
# -----------------------------
vecs = embed_texts(query_ingredients, model)

# -----------------------------
# Search FAISS
# -----------------------------
faiss_results = []
for vec in vecs:
    results = handler.search(vec, column_key="names", top_k=TOP_K)
    faiss_results.extend(results)

# -----------------------------
# Aggregate results theo món + token-level match
# -----------------------------
agg_results = {}
for r in faiss_results:
    rowid = r["_rowid"]
    dish_name = r["dish_name"]
    ingredients = r.get("ingredient_names", [])

    # flatten và canonical token
    flat_tokens = []
    for item in ingredients:
        if isinstance(item, list):
            for i in item:
                flat_tokens.extend(tokenize_canonical(i))
        else:
            flat_tokens.extend(tokenize_canonical(str(item)))

    # match ratio token-level
    match_count = sum(1 for qt in query_tokens if qt in flat_tokens)
    match_ratio = match_count / len(query_tokens)

    # final score
    final_score = ALPHA * r["_distance"] + BETA * match_ratio

    if rowid not in agg_results:
        agg_results[rowid] = {
            "dish_name": dish_name,
            "ingredients": ingredients,
            "cosine": r["_distance"],
            "match_ratio": match_ratio,
            "final_score": final_score
        }
    else:
        if final_score > agg_results[rowid]["final_score"]:
            agg_results[rowid]["cosine"] = r["_distance"]
            agg_results[rowid]["match_ratio"] = match_ratio
            agg_results[rowid]["final_score"] = final_score

# -----------------------------
# Lấy top FINAL_TOP_K món
# -----------------------------
final_sorted = sorted(agg_results.values(), key=lambda x: x["final_score"], reverse=True)[:FINAL_TOP_K]

# -----------------------------
# In kết quả
# -----------------------------
print("\n🎯 Top gợi ý món ăn:")
for r in final_sorted:
    # flatten thành string
    flat_ing = []
    for item in r["ingredients"]:
        if isinstance(item, list):
            flat_ing.extend(item)
        else:
            flat_ing.append(str(item))
    ingredients_str = ", ".join(flat_ing)

    print(f"- {r['dish_name']}  |  final_score={r['final_score']:.4f}")
    print(f"  Ingredients: {ingredients_str}")
    print(f"  Cosine: {r['cosine']:.4f} | Match ratio: {r['match_ratio']:.2f}\n")
