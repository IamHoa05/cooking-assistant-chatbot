# import pandas as pd
# import numpy as np
# from faiss_handler import FAISSHandler
# from embedder import load_vietnamese_embedding_model, embed_texts

# if __name__ == "__main__":
#     # 1. Load DataFrame thật đã embed
#     print("🚀 Loading DataFrame...")
#     df = pd.read_pickle("./data/recipes_embeddings.pkl")

#     # 2. Map key → tên cột embedding
#     embedding_columns = {
#         "names": "ingredient_names_embedding",
#         "quantities": "ingredient_quantities_embedding",
#         "dish": "dish_name_embedding",
#     }

#     # 3. Load FAISS handler
#     print("🔍 Loading FAISS indexes...")
#     handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="./faiss_indexes")
#     print("✅ FAISS handler loaded!")

#     # 4. Nhập danh sách nguyên liệu từ user
#     user_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
#     ingredients = [x.strip() for x in user_input.split(",") if x.strip()]
    
#     # 5. Load embedding model
#     print("🧠 Loading embedding model (BAAI/bge-base-en-v1.5)...")
#     model = load_vietnamese_embedding_model(device="cpu")

#     # 6. Encode input nguyên liệu (query)
#     print("\n🔧 Encoding ingredients...")
#     query_vecs = embed_texts([", ".join(ingredients)], model, text_type="query")
#     query_vec = np.array(query_vecs[0]).astype("float32")

#     # 7. Search FAISS
#     print(f"🔎 Searching FAISS index ({embedding_columns['names']})...\n")
#     results = handler.search(query_vector=query_vec, column_key="names", top_k=5)

#     # 8. In kết quả
#     print("\n==============================")
#     print("🍽️ TOP RESULTS:")
#     for i, r in enumerate(results, 1):
#         print(f"{i}. Dish: {r.get('dish_name')}  | distance = {r['_distance']:.4f}")
#         print(f"   Ingredients: {r.get('ingredient_names')}")


# import pandas as pd
# import numpy as np
# from faiss_handler import FAISSHandler
# from embedder import load_vietnamese_embedding_model, embed_texts
# from collections import defaultdict

# if __name__ == "__main__":
#     print("🚀 Loading DataFrame...")
#     df = pd.read_pickle("./data/recipes_embeddings.pkl")
#     # print(df.columns)
#     embedding_columns = {
#         "names": "ingredient_names_embedding",
#         "quantities": "ingredient_quantities_embedding",
#         "dish": "dish_name_embedding",
#     }

#     print("🔍 Loading FAISS indexes...")
#     handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="./faiss_indexes")
#     print("✅ FAISS handler loaded!")

#     user_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
#     ingredients = [x.strip() for x in user_input.split(",") if x.strip()]

#     print("🧠 Loading embedding model (BAAI/bge-base-en-v1.5)...")
#     model = load_vietnamese_embedding_model(device="cpu")

#     print("\n🔧 Encoding từng nguyên liệu...")
#     ingredient_vecs = embed_texts(ingredients, model, text_type="query")
#     ingredient_vecs = [np.array(v).astype("float32") for v in ingredient_vecs]

#     print(f"\n🔎 Searching FAISS index for từng nguyên liệu...\n")

#     # 👉 Score aggregator
#     score_map = defaultdict(float)
#     count_map = defaultdict(int)

#     for ing, vec in zip(ingredients, ingredient_vecs):
#         print(f"\n🔹 Tìm theo nguyên liệu: {ing}")
#         results = handler.search(query_vector=vec, column_key="names", top_k=20)

#         for r in results:
#             row_id = r["__rowid__"]     # 🔥 index gốc
#             score = -r["_distance"]

#             score_map[row_id] += score
#             count_map[row_id] += 1

#     # 👉 Ranking theo tổng score + số nguyên liệu match
#     ranked = sorted(score_map.items(), key=lambda x: (count_map[x[0]], x[1]), reverse=True)

#     print("\n==============================")
#     print("🍽️ TOP RESULTS:")

#     top = 5
#     for i, (row_id, total_score) in enumerate(ranked[:top], 1):
#         row = df.iloc[row_id]   # 🔥 lấy lại dòng thật

#         print(f"{i}. Dish: {row['dish_name']} | matched {count_map[row_id]}/{len(ingredients)}")
#         print(f"   Ingredients: {row['ingredient_names']}\n")


# # test_faiss_cosine_fuzzy.py
# import pandas as pd
# import numpy as np
# from faiss_handler import FAISSHandler
# from embedder import load_vietnamese_embedding_model, embed_texts
# from collections import defaultdict
# import unicodedata
# from difflib import get_close_matches

# # -----------------------------
# # Helpers
# # -----------------------------
# def clean_ingredient(text):
#     text = text.lower().strip()
#     text = ''.join(c for c in unicodedata.normalize('NFD', text)
#                    if unicodedata.category(c) != 'Mn')
#     return text

# def fuzzy_match(recipe_ings, input_ings, cutoff=0.6):
#     recipe_clean = [clean_ingredient(i) for i in recipe_ings]
#     input_clean = [clean_ingredient(i) for i in input_ings]

#     matched = []
#     for ing in input_clean:
#         close = get_close_matches(ing, recipe_clean, n=1, cutoff=cutoff)
#         if close:
#             matched.append(close[0])
#     match_count = len(matched)
#     match_ratio = match_count / len(input_clean) if input_clean else 0
#     return matched, match_count, match_ratio

# def avg_cosine_score(input_vecs, recipe_vecs):
#     """Tính trung bình max cosine similarity giữa các nguyên liệu input và món"""
#     scores = []
#     for v_in in input_vecs:
#         sims = [np.dot(v_in, v_rec)/(np.linalg.norm(v_in)*np.linalg.norm(v_rec)) for v_rec in recipe_vecs]
#         scores.append(max(sims))
#     return np.mean(scores)

# # -----------------------------
# # Main search
# # -----------------------------
# def search_dishes_with_cosine_fuzzy(df, handler, input_ingredients, top_faiss=100, top_k=5, alpha=0.7):
#     """
#     Tìm món ăn dựa trên cosine similarity + fuzzy match
#     alpha: trọng số cosine, 1-alpha: trọng số fuzzy
#     """
#     print("🧠 Encoding input ingredients...")
#     model = load_vietnamese_embedding_model(device="cpu")
#     input_vecs = embed_texts(input_ingredients, model, text_type="query")
#     input_vecs = [np.array(v).astype("float32") for v in input_vecs]

#     # 1️⃣ FAISS search từng nguyên liệu → gộp score về món
#     score_map = defaultdict(list)  # idx món -> list vec của nguyên liệu món
#     for vec in input_vecs:
#         results = handler.search(query_vector=vec, column_key="names", top_k=top_faiss)
#         for r in results:
#             idx = r.get("_rowid__") or r.get("index")
#             if idx is not None:
#                 score_map[idx].append(r["_distance"])  # lưu score embedding (cosine)

#     # 2️⃣ Tính score tổng hợp
#     final_results = []
#     for idx, row in df.iloc[list(score_map.keys())].iterrows():
#         # embedding món
#         recipe_vecs = []
#         ing_embeds = row["ingredient_names_embedding"]
#         if isinstance(ing_embeds, list):
#             for vec in ing_embeds:
#                 recipe_vecs.append(np.array(vec).astype("float32"))
#         if not recipe_vecs:
#             continue

#         score_cosine = avg_cosine_score(input_vecs, recipe_vecs)
#         matched, match_count, match_ratio = fuzzy_match(row['ingredient_names'], input_ingredients)
#         # tổng hợp
#         score_total = alpha*score_cosine + (1-alpha)*match_ratio

#         final_results.append({
#             "dish_name": row['dish_name'],
#             "ingredient_names": row['ingredient_names'],
#             "matched_ingredients": matched,
#             "match_count": match_count,
#             "match_ratio": match_ratio,
#             "score_cosine": score_cosine,
#             "score_total": score_total
#         })

#     # 3️⃣ Sắp xếp theo score_total giảm dần
#     final_results = sorted(final_results, key=lambda x: x["score_total"], reverse=True)

#     return final_results[:top_k]

# # -----------------------------
# # Example usage
# # -----------------------------
# if __name__ == "__main__":
#     df = pd.read_pickle("./data/recipes_embeddings.pkl")
#     print(df.columns)
#     embedding_columns = {
#         "names": "ingredient_names_embedding",
#         "quantities": "ingredient_quantities_embedding",
#         "dish": "dish_name_embedding",
#     }
#     handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="./faiss_indexes")

#     user_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
#     input_ingredients = [x.strip() for x in user_input.split(",") if x.strip()]

#     results = search_dishes_with_cosine_fuzzy(df, handler, input_ingredients, top_faiss=100, top_k=5, alpha=0.7)

#     print("\n==============================")
#     print("🍽️ TOP RESULTS:")
#     for i, r in enumerate(results, 1):
#         print(f"{i}. Dish: {r['dish_name']} | matched {r['match_count']}/{len(input_ingredients)}")
#         print(f"   Ingredients: {r['ingredient_names']}")
#         print(f"   Matched ingredients: {r['matched_ingredients']}")
#         print(f"   Cosine score: {r['score_cosine']:.4f}")
#         print(f"   Total score: {r['score_total']:.4f}\n")


# test_faiss_cosine_fuzzy.py
import pandas as pd
import numpy as np
from faiss_handler import FAISSHandler
from embedder import load_vietnamese_embedding_model, embed_texts
from collections import defaultdict
import unicodedata
from difflib import get_close_matches
import ast

# -----------------------------
# Helpers
# -----------------------------
def clean_ingredient(text):
    """Chuẩn hóa nguyên liệu: lowercase, remove dấu, trim"""
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    return text

def parse_ingredient_list(val):
    """Chuyển string dạng list thành list thực sự"""
    if isinstance(val, list):
        return val
    elif isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]
    return []

def fuzzy_match_debug(recipe_ings, input_ings, cutoff=0.6):
    """
    Fuzzy match với debug: trả về matched, match_count, match_ratio, matched_pairs
    matched_pairs = list of tuples (input_ing, recipe_ing_matched or None)
    """
    recipe_ings = parse_ingredient_list(recipe_ings)
    recipe_clean = [clean_ingredient(i) for i in recipe_ings]
    input_clean = [clean_ingredient(i) for i in input_ings]

    matched = []
    matched_pairs = []
    for ing_in, ing_clean in zip(input_ings, input_clean):
        close = get_close_matches(ing_clean, recipe_clean, n=1, cutoff=cutoff)
        if close:
            matched.append(close[0])
            matched_pairs.append((ing_in, close[0]))
        else:
            matched_pairs.append((ing_in, None))

    match_count = len(matched)
    match_ratio = match_count / len(input_ings) if input_ings else 0
    return matched, match_count, match_ratio, matched_pairs

def avg_cosine_score(input_vecs, recipe_vecs):
    """Tính trung bình max cosine similarity giữa các nguyên liệu input và món"""
    scores = []
    for v_in in input_vecs:
        sims = [np.dot(v_in, v_rec)/(np.linalg.norm(v_in)*np.linalg.norm(v_rec)) for v_rec in recipe_vecs]
        scores.append(max(sims))
    return np.mean(scores)

# -----------------------------
# Main search
# -----------------------------
def search_dishes_with_cosine_fuzzy(df, handler, input_ingredients, top_faiss=100, top_k=5, alpha=0.7):
    """
    Tìm món ăn dựa trên cosine similarity + fuzzy match
    alpha: trọng số cosine, 1-alpha: trọng số fuzzy
    """
    print("🧠 Encoding input ingredients...")
    model = load_vietnamese_embedding_model(device="cpu")
    input_vecs = embed_texts(input_ingredients, model, text_type="query")
    input_vecs = [np.array(v).astype("float32") for v in input_vecs]

    # 1️⃣ FAISS search từng nguyên liệu → gộp score về món
    score_map = defaultdict(list)  # idx món -> list vec embedding món
    for vec in input_vecs:
        results = handler.search(query_vector=vec, column_key="names", top_k=top_faiss)
        for r in results:
            idx = r.get("_rowid__") or r.get("index")
            if idx is not None:
                score_map[idx].append(r["_distance"])  # cosine similarity

    # 2️⃣ Tính score tổng hợp + fuzzy
    final_results = []
    for idx, row in df.iloc[list(score_map.keys())].iterrows():
        # embedding món
        recipe_vecs = []
        ing_embeds = row["ingredient_names_embedding"]
        if isinstance(ing_embeds, list):
            for vec in ing_embeds:
                recipe_vecs.append(np.array(vec).astype("float32"))
        if not recipe_vecs:
            continue

        score_cosine = avg_cosine_score(input_vecs, recipe_vecs)
        matched, match_count, match_ratio, matched_pairs = fuzzy_match_debug(row['ingredient_names'], input_ingredients)
        score_total = alpha*score_cosine + (1-alpha)*match_ratio

        # Debug in
        print("\nRecipe raw:", row['ingredient_names'])
        print("Recipe cleaned:", parse_ingredient_list(row['ingredient_names']))
        print("Input cleaned:", [clean_ingredient(i) for i in input_ingredients])
        print("Fuzzy debug pairs (input → recipe match):", matched_pairs)
        print("Cosine score:", score_cosine)
        print("Fuzzy match ratio:", match_ratio)
        print("Total score:", score_total)

        final_results.append({
            "dish_name": row['dish_name'],
            "ingredient_names": parse_ingredient_list(row['ingredient_names']),
            "matched_ingredients": matched,
            "match_count": match_count,
            "match_ratio": match_ratio,
            "score_cosine": score_cosine,
            "score_total": score_total
        })

    # 3️⃣ Sắp xếp theo score_total giảm dần
    final_results = sorted(final_results, key=lambda x: x["score_total"], reverse=True)
    return final_results[:top_k]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    df = pd.read_pickle("./data/recipes_embeddings.pkl")
    embedding_columns = {
        "names": "ingredient_names_embedding",
        "quantities": "ingredient_quantities_embedding",
        "dish": "dish_name_embedding",
    }
    handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="./faiss_indexes")

    user_input = input("💬 Nhập danh sách nguyên liệu (ngăn cách bằng dấu phẩy):\n👉 Ingredients: ")
    input_ingredients = [x.strip() for x in user_input.split(",") if x.strip()]

    results = search_dishes_with_cosine_fuzzy(df, handler, input_ingredients, top_faiss=100, top_k=5, alpha=0.7)

    print("\n==============================")
    print("🍽️ TOP RESULTS:")
    for i, r in enumerate(results, 1):
        print(f"{i}. Dish: {r['dish_name']} | matched {r['match_count']}/{len(input_ingredients)}")
        print(f"   Ingredients: {r['ingredient_names']}")
        print(f"   Matched ingredients: {r['matched_ingredients']}")
        print(f"   Cosine score: {r['score_cosine']:.4f}")
        print(f"   Total score: {r['score_total']:.4f}\n")
