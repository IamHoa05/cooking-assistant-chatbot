# search_engine.py
import numpy as np
import pandas as pd
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
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]
    return []

def canonicalize_ingredient_list(col):
    """
    Chuẩn hóa danh sách nguyên liệu nhưng GIỮ NGUYÊN DẤU.
    Trả về list các nguyên liệu ở dạng:
    - list thực sự
    - lowercase
    - đã trim
    """
    result = []

    for x in col:
        # 1. Nếu đã là list → chuẩn hóa từng phần tử
        if isinstance(x, list):
            val = [v.strip().lower() for v in x if isinstance(v, str) and v.strip()]
            result.append(val)
            continue

        # 2. Nếu là string dạng list → convert bằng ast
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
                pass  # fallback to split

            # 3. Nếu string không phải list → split bằng dấu phẩy
            val = [v.strip().lower() for v in x.split(",") if v.strip()]
            result.append(val)
            continue

        # 4. Nếu dữ liệu rác → trả list rỗng
        result.append([])

    return result



def fuzzy_match(recipe_ings, input_ings, cutoff=0.8):
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

def compute_scores(recipe_ing, query_ing):
    """Tính exact match và jaccard score"""
    recipe_set = set(recipe_ing)
    query_set = set(query_ing)
    # Exact match ratio
    exact = len(recipe_set & query_set) / len(query_set) if query_set else 0
    # Jaccard index
    jaccard = len(recipe_set & query_set) / len(recipe_set | query_set) if recipe_set else 0
    return exact, jaccard

# -----------------------------
# Main search function (API dùng hàm này)
# -----------------------------
def search_dishes(df, handler, input_ingredients, top_faiss=100, top_k=5):
    from app.utils.new_embedder import load_embedding_model, embed_texts

    # encode input
    model = load_embedding_model("BAAI/bge-m3")
    query_vec = embed_texts(input_ingredients, model)
    query_vec = np.mean(query_vec, axis=0)

    # search FAISS
    results = handler.search(query_vec=query_vec, column_key="names", top_k=top_faiss)

    # tính scores
    for r in results:
        # giờ r["ingredient_names"] đã tồn tại
        exact, jaccard = compute_scores(r["ingredient_names"], input_ingredients)
        fuzzy = fuzzy_match(r["ingredient_names"], input_ingredients)
        r["exact_match"] = exact
        r["jaccard"] = jaccard
        r["fuzzy_match"] = fuzzy

        # final score
        r["final_score"] = (
            r["_distance"] * 0.7 +
            exact * 0.15 +
            jaccard * 0.1 +
            fuzzy * 0.05
        )

        # bonus theo số nguyên liệu match
        matched_count = sum(1 for ing in input_ingredients if ing in r["ingredient_names"])
        if matched_count >= 1:
            bonus = 0.05 * min(matched_count, 4)
            r["final_score"] *= (1 + bonus)

    # sắp xếp theo final_score
    final_results = sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]

    # format output
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
# Hàm khởi tạo và chạy search
# -----------------------------
def initialize_search_engine(data_path="C:/Users/thuyt/Desktop/Artificial Intelligence/cooking-assistant-chatbot/backend/app/utils/new_data/recipes_embeddings.pkl", 
                           index_dir="C:/Users/thuyt/Desktop/Artificial Intelligence/cooking-assistant-chatbot/backend/app/utils/new_faiss_indexes"):
    """Khởi tạo search engine với data và FAISS index"""
    # Load DataFrame
    df = pd.read_pickle(data_path)
    df["ingredient_names"] = canonicalize_ingredient_list(df["ingredient_names"])
    
    # Load FAISS handler - THÊM XỬ LÝ LỖI
    from app.utils.new_faiss_handler import FAISSHandler
    
    try:
        # Thử không có embedding_columns
        handler = FAISSHandler(df, index_dir=index_dir)
    except TypeError as e:
        # Nếu lỗi, thử với embedding_columns
        try:
            embedding_columns = {
                "names": "ingredient_names_embedding",
                "quantities": "ingredient_quantities_embedding",
                "dish": "dish_name_embedding"
            }
            handler = FAISSHandler(df, embedding_columns=embedding_columns, index_dir=index_dir)
        except Exception as e:
            print(f"❌ Lỗi khởi tạo FAISSHandler: {e}")
            raise
    
    return df, handler

def search_by_ingredients(input_ingredients, df, handler, top_k=5):
    """
    Hàm chính để search món ăn theo nguyên liệu
    Input: list nguyên liệu (string)
    Output: list kết quả với đầy đủ thông tin
    """
    if isinstance(input_ingredients, str):
        # Nếu input là string, split bằng comma
        input_ingredients = [x.strip().lower() for x in input_ingredients.split(",") if x.strip()]
    
    return search_dishes(df, handler, input_ingredients, top_k=top_k)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Khởi tạo search engine
    df, handler = initialize_search_engine()
    
    # Test search
    user_input = "ức gà, bắp cải, khoai tây, cà rốt, bí đỏ"
    
    results = search_by_ingredients(user_input, df, handler, top_k=5)
    
    print(f"🔍 Kết quả tìm kiếm cho: {user_input}")
    print("=" * 60)
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['dish_name']} | Score: {r['final_score']:.4f}")
        print(f"   Nguyên liệu: {', '.join(r['ingredients'])}")
        print(f"   Chi tiết: Cosine={r['score_breakdown']['cosine']:.3f}, "
              f"Exact={r['score_breakdown']['exact_match']:.3f}, "
              f"Jaccard={r['score_breakdown']['jaccard']:.3f}, "
              f"Fuzzy={r['score_breakdown']['fuzzy']:.3f}")
        print()