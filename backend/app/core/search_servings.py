# app/core/search_servings.py
from typing import List, Dict
import numpy as np

def search_dishes_by_servings(df, faiss_handler, target_servings: int, top_k: int = 5):
    """
    Tìm món ăn gần nhất với số khẩu phần target_servings.
    Dựa trên cột 'servings' trong DataFrame.
    """
    results = []

    for idx, row in df.iterrows():
        servings = row.get("servings")
        if servings is None:
            continue
        distance = abs(servings - target_servings)
        results.append((distance, row))

    results.sort(key=lambda x: x[0])
    top_rows = [r[1] for r in results[:top_k]]

    # Trả về danh sách dict gồm dish_name và score (distance)
    return [{"dish_name": r["dish_name"], "score": 1/(1 + abs(r.get("servings",0)-target_servings))} for r in top_rows]
