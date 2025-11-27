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
def search_dishes(df, handler, input_ingredients,
                  top_faiss=100, top_k=5, alpha=0.7):
    """
    ĐÃ SỬA:
    - Không dùng embedding, không gọi FAISS
    - Chỉ dùng fuzzy_match + keyword match
    - Giữ nguyên TÊN HÀM để không phá API
    """
    if not input_ingredients:
        return []

    # Chuẩn hóa input
    input_clean = [clean_ingredient(i) for i in input_ingredients]

    results = []

    for idx, row in df.iterrows():
        recipe_ings = parse_ingredient_list(row["ingredient_names"])
        recipe_clean = [clean_ingredient(i) for i in recipe_ings]

        # Fuzzy
        score_fuzzy = fuzzy_match(recipe_ings, input_ingredients)

        # Keyword match (tăng độ chính xác)
        keyword_hits = sum(1 for x in input_clean if x in recipe_clean)
        score_keyword = keyword_hits / len(input_clean)

        # Tổng hợp (embedding bỏ → cosine = 0)
        score_total = 0.7 * score_keyword + 0.3 * score_fuzzy

        results.append((row["dish_name"], score_total))

    # Sắp xếp
    results.sort(key=lambda x: x[1], reverse=True)

    return [name for name, score in results[:top_k]]
