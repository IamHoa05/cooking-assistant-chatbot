# app/core/search_difficulty.py
"""
Module tìm kiếm món ăn theo độ khó (difficulty) trên DataFrame.
"""

import pandas as pd


def get_dishes_by_difficulty(df: pd.DataFrame, difficulty: str, top_k: int = 5):
    """
    Lấy danh sách món ăn theo độ khó.

    Args:
        df (pd.DataFrame): DataFrame chứa ít nhất các cột:
            - 'dish_name': tên món ăn
            - 'difficulty': độ khó món ăn ('easy', 'medium', 'hard')
        difficulty (str): độ khó cần lọc (không phân biệt chữ hoa/thường)
        top_k (int): số lượng kết quả tối đa trả về

    Returns:
        List[dict]: danh sách dict, mỗi dict chứa ít nhất:
            - 'dish_name'
            - 'difficulty'
    """
    # Chuẩn hóa độ khó
    difficulty_lower = difficulty.strip().lower()

    # Lọc DataFrame theo độ khó
    df_filtered = df[df["difficulty"].str.lower() == difficulty_lower]

    # Lấy tối đa top_k món
    top_dishes = df_filtered.head(top_k).to_dict(orient="records")

    return top_dishes