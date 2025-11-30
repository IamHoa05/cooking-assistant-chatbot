# app/core/search_servings.py

import numpy as np
import pandas as pd

def search_dishes_by_servings(df: pd.DataFrame, faiss_handler, target_servings: int, top_k: int = 5):
    """
    Tìm món ăn gần đúng với số khẩu phần mục tiêu.

    Args:
        df (pd.DataFrame): DataFrame chứa recipes, cần có cột 'dish_name' và 'servings'
        faiss_handler: chưa sử dụng trực tiếp ở đây, giữ để tương thích với pipeline
        target_servings (int): số khẩu phần người dùng muốn
        top_k (int): số món trả về tối đa

    Returns:
        List[dict]: danh sách món ăn, mỗi món gồm:
            - dish_name: tên món
            - score: điểm gần đúng với khẩu phần (càng gần càng cao)
    """
    results = []

    for idx, row in df.iterrows():
        servings = row.get("servings")
        if servings is None:
            continue

        # Tính khoảng cách tuyệt đối
        distance = abs(servings - target_servings)
        results.append((distance, row))

    # Sắp xếp theo khoảng cách nhỏ → gần số target nhất
    results.sort(key=lambda x: x[0])

    # Lấy top_k món
    top_rows = [r[1] for r in results[:top_k]]

    # Chuyển thành list dict với điểm "score" = 1 / (1 + distance)
    formatted_results = [
        {
            "dish_name": r["dish_name"],
            "score": 1 / (1 + abs(r.get("servings", 0) - target_servings))
        }
        for r in top_rows
    ]

    return formatted_results
