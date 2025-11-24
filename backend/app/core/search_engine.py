# search_engine.py
import numpy as np
import pandas as pd
from collections import defaultdict
import unicodedata
from difflib import get_close_matches
import ast

from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
from app.utils.faiss_handler import FAISSHandler


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
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]
    return []


def fuzzy_match(recipe_ings, input_ings, cutoff=0.6):
    """Fuzzy match mức tối thiểu → trả về match_ratio"""
    recipe_ings = parse_ingredient_list(recipe_ings)
    recipe_clean = [clean_ingredient(i) for i in recipe_ings]
    input_clean = [clean_ingredient(i) for i in input_ings]

    matched = 0
    for ing in input_clean:
        close = get_close_matches(ing, recipe_clean, n=1, cutoff=cutoff)
        if close:
            matched += 1

    match_ratio = matched / len(input_ings) if input_ings else 0
    return match_ratio


def avg_cosine_score(input_vecs, recipe_vecs):
    """Trung bình cosine similarity (max theo từng nguyên liệu input)."""
    scores = []
    for v_in in input_vecs:
        sims = [
            np.dot(v_in, v_rec) / (np.linalg.norm(v_in) * np.linalg.norm(v_rec))
            for v_rec in recipe_vecs
        ]
        scores.append(max(sims))
    return np.mean(scores)


# -----------------------------
# Main search function (API dùng hàm này)
# -----------------------------
def search_dishes(df, handler: FAISSHandler, input_ingredients,
                  top_faiss=100, top_k=5, alpha=0.7):
    """
    Tìm món ăn dựa trên:
    - cosine similarity
    - fuzzy matching
    - tổng hợp score_total = alpha*cosine + (1-alpha)*fuzzy
    Trả về: list tên món ăn
    """
    if not input_ingredients:
        return []

    # 1️⃣ Encode input ingredients
    model = load_vietnamese_embedding_model(device="cpu")
    input_vecs = embed_texts(input_ingredients, model, text_type="query")
    input_vecs = [np.array(v).astype("float32") for v in input_vecs]

    # 2️⃣ FAISS search từng nguyên liệu
    score_map = defaultdict(list)

    for vec in input_vecs:
        results = handler.search(query_vector=vec, column_key="names", top_k=top_faiss)
        for r in results:
            idx = r.get("_rowid__") or r.get("index")
            if idx is not None:
                score_map[idx].append(r["_distance"])

    if not score_map:
        return []

    # 3️⃣ Tính tổng score
    final_scores = []

    for idx, row in df.iloc[list(score_map.keys())].iterrows():
        # lấy embedding của món
        recipe_vecs = []
        embeds = row["ingredient_names_embedding"]

        if isinstance(embeds, list):
            recipe_vecs = [np.array(v).astype("float32") for v in embeds]

        if not recipe_vecs:
            continue

        score_cos = avg_cosine_score(input_vecs, recipe_vecs)
        score_fuzzy = fuzzy_match(row["ingredient_names"], input_ingredients)

        score_total = alpha * score_cos + (1 - alpha) * score_fuzzy

        final_scores.append((row["dish_name"], score_total))

    # 4️⃣ Sắp xếp theo score giảm dần
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # 5️⃣ Return danh sách tên món ăn
    return [name for name, score in final_scores[:top_k]]
