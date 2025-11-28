# # search_engine.py
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# import unicodedata
# from difflib import get_close_matches
# import ast

# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# from app.utils.faiss_handler import FAISSHandler


# # -----------------------------
# # Helpers
# # -----------------------------
# def clean_ingredient(text):
#     """Chuẩn hóa nguyên liệu: lowercase, remove dấu, trim"""
#     text = text.lower().strip()
#     text = ''.join(c for c in unicodedata.normalize('NFD', text)
#                    if unicodedata.category(c) != 'Mn')
#     return text


# def parse_ingredient_list(val):
#     """Chuyển string dạng list thành list thực sự"""
#     if isinstance(val, list):
#         return val
#     if isinstance(val, str):
#         try:
#             return ast.literal_eval(val)
#         except Exception:
#             return [val]
#     return []


# def fuzzy_match(recipe_ings, input_ings, cutoff=0.6):
#     """Fuzzy match mức tối thiểu → trả về match_ratio"""
#     recipe_ings = parse_ingredient_list(recipe_ings)
#     recipe_clean = [clean_ingredient(i) for i in recipe_ings]
#     input_clean = [clean_ingredient(i) for i in input_ings]

#     matched = 0
#     for ing in input_clean:
#         close = get_close_matches(ing, recipe_clean, n=1, cutoff=cutoff)
#         if close:
#             matched += 1

#     match_ratio = matched / len(input_ings) if input_ings else 0
#     return match_ratio


# def avg_cosine_score(input_vecs, recipe_vecs):
#     """Trung bình cosine similarity (max theo từng nguyên liệu input)."""
#     scores = []
#     for v_in in input_vecs:
#         sims = [
#             np.dot(v_in, v_rec) / (np.linalg.norm(v_in) * np.linalg.norm(v_rec))
#             for v_rec in recipe_vecs
#         ]
#         scores.append(max(sims))
#     return np.mean(scores)


# # -----------------------------
# # Main search function (API dùng hàm này)
# # -----------------------------
# def search_dishes(df, handler: FAISSHandler, input_ingredients,
#                   top_faiss=100, top_k=5, alpha=0.7):
#     """
#     Tìm món ăn dựa trên:
#     - cosine similarity
#     - fuzzy matching
#     - tổng hợp score_total = alpha*cosine + (1-alpha)*fuzzy
#     Trả về: list tên món ăn
#     """
#     if not input_ingredients:
#         return []

#     # 1️⃣ Encode input ingredients
#     model = load_vietnamese_embedding_model(device="cpu")
#     input_vecs = embed_texts(input_ingredients, model, text_type="query")
#     input_vecs = [np.array(v).astype("float32") for v in input_vecs]

#     # 2️⃣ FAISS search từng nguyên liệu
#     score_map = defaultdict(list)

#     for vec in input_vecs:
#         results = handler.search(query_vector=vec, column_key="names", top_k=top_faiss)
#         for r in results:
#             idx = r.get("_rowid__") or r.get("index")
#             if idx is not None:
#                 score_map[idx].append(r["_distance"])

#     if not score_map:
#         return []

#     # 3️⃣ Tính tổng score
#     final_scores = []

#     for idx, row in df.iloc[list(score_map.keys())].iterrows():
#         # lấy embedding của món
#         recipe_vecs = []
#         embeds = row["ingredient_names_embedding"]

#         if isinstance(embeds, list):
#             recipe_vecs = [np.array(v).astype("float32") for v in embeds]

#         if not recipe_vecs:
#             continue

#         score_cos = avg_cosine_score(input_vecs, recipe_vecs)
#         score_fuzzy = fuzzy_match(row["ingredient_names"], input_ingredients)

#         score_total = alpha * score_cos + (1 - alpha) * score_fuzzy

#         final_scores.append((row["dish_name"], score_total))

#     # 4️⃣ Sắp xếp theo score giảm dần
#     final_scores.sort(key=lambda x: x[1], reverse=True)

#     # 5️⃣ Return danh sách tên món ăn
#     return [name for name, score in final_scores[:top_k]]

# search_engine.py
import os
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
import unicodedata
from difflib import get_close_matches
import ast

from ..utils.faiss_handler import FAISSHandler
from ..utils.embedder import load_embedding_model, embed_texts

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.yml"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

EMBED_MODEL_NAME = CONFIG["embedding"]["model_name"]
EMBED_BATCH_SIZE = CONFIG["embedding"].get("batch_size", 32)

# -----------------------------
# Helpers
# -----------------------------
def clean_ingredient(text):
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    return text

def parse_ingredient_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]
    return []

def canonicalize_ingredient_list(col):
    result = []

    for x in col:
        if isinstance(x, list):
            val = [v.strip().lower() for v in x if isinstance(v, str) and v.strip()]
            result.append(val)
            continue

        if isinstance(x, str):
            x = x.strip()
            if not x:
                result.append([])
                continue

            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    val = [v.strip().lower() for v in parsed if isinstance(v, str) and v.strip()]
                    result.append(val)
                    continue
            except:
                pass

            val = [v.strip().lower() for v in x.split(",") if v.strip()]
            result.append(val)
            continue

        result.append([])

    return result

def fuzzy_match(recipe_ings, input_ings, cutoff=0.8):
    recipe_ings = parse_ingredient_list(recipe_ings)
    recipe_clean = [clean_ingredient(i) for i in recipe_ings]
    input_clean = [clean_ingredient(i) for i in input_ings]

    matched = 0
    for ing in input_clean:
        close = get_close_matches(ing, recipe_clean, n=1, cutoff=cutoff)
        if close:
            matched += 1

    return matched / len(input_ings) if input_ings else 0

def compute_scores(recipe_ing, query_ing):
    recipe_set = set(recipe_ing)
    query_set = set(query_ing)
    exact = len(recipe_set & query_set) / len(query_set) if query_set else 0
    jaccard = len(recipe_set & query_set) / len(recipe_set | query_set) if recipe_set else 0
    return exact, jaccard

# -----------------------------
# Main search
# -----------------------------
def search_dishes(df, handler, input_ingredients, top_faiss=100, top_k=5):
    # encode input using model from config
    model = load_embedding_model(EMBED_MODEL_NAME)
    query_vec = embed_texts(input_ingredients, model, batch_size=EMBED_BATCH_SIZE)
    query_vec = np.mean(query_vec, axis=0)

    results = handler.search(query_vec=query_vec, column_key="names", top_k=top_faiss)

    for r in results:
        exact, jaccard = compute_scores(r["ingredient_names"], input_ingredients)
        fuzzy = fuzzy_match(r["ingredient_names"], input_ingredients)
        r["exact_match"] = exact
        r["jaccard"] = jaccard
        r["fuzzy_match"] = fuzzy

        r["final_score"] = (
            r["_distance"] * 0.7 +
            exact * 0.15 +
            jaccard * 0.1 +
            fuzzy * 0.05
        )

        matched_count = sum(1 for ing in input_ingredients if ing in r["ingredient_names"])
        if matched_count >= 1:
            bonus = 0.05 * min(matched_count, 4)
            r["final_score"] *= (1 + bonus)

    final_results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]

    formatted_results = []
    for r in final_results:
        matched_count = sum(1 for ing in input_ingredients if ing in r["ingredient_names"])
        formatted_results.append({
            "dish_name": r["dish_name"],
            "ingredients": r["ingredient_names"],
            "final_score": round(r["final_score"], 4),
            "score_breakdown": {
                "cosine": round(r["_distance"], 4),
                "exact_match": round(r["exact_match"], 4),
                "jaccard": round(r["jaccard"], 4),
                "fuzzy": round(r["fuzzy_match"], 4),
                "matched_count": matched_count
            }
        })

    return formatted_results

# -----------------------------
# Initialize engine
# -----------------------------
def initialize_search_engine(
    data_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/data/recipes_embeddings.pkl")),
    index_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/faiss_indexes"))
):
    df = pd.read_pickle(data_path)
    df["ingredient_names"] = canonicalize_ingredient_list(df["ingredient_names"])

    handler = FAISSHandler(df, index_dir=index_dir)
    return df, handler

def search_by_ingredients(input_ingredients, df, handler, top_k=5):
    if isinstance(input_ingredients, str):
        input_ingredients = [x.strip().lower() for x in input_ingredients.split(",") if x.strip()]
    return search_dishes(df, handler, input_ingredients, top_k=top_k)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df, handler = initialize_search_engine()
    user_input = "ức gà, bắp cải, khoai tây, cà rốt, bí đỏ"
    results = search_by_ingredients(user_input, df, handler, top_k=5)

    print(f"🔍 Kết quả tìm kiếm cho: {user_input}")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['dish_name']} | Score: {r['final_score']:.4f}")
        print(f"   Nguyên liệu: {', '.join(r['ingredients'])}")
        print(f"   Chi tiết: Cosine={r['score_breakdown']['cosine']:.3f}, "
              f"Exact={r['score_breakdown']['exact_match']:.3f}, "
              f"Jaccard={r['score_breakdown']['jaccard']:.3f}, "
              f"Fuzzy={r['score_breakdown']['fuzzy']:.3f}")
        print()
