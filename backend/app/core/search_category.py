# app/core/search_category.py
"""
Module tìm kiếm món ăn theo category trực tiếp trên DataFrame.
Không sử dụng metadata hay embeddings.
"""

import pandas as pd


def search_dishes_by_category(df: pd.DataFrame, category: str, max_results: int = 5):
    """
    Tìm món ăn theo thể loại (category) trên DataFrame.

    Args:
        df (pd.DataFrame): DataFrame chứa ít nhất 2 cột:
            - 'dish_name': tên món ăn
            - 'category': thể loại món ăn
        category (str): thể loại món ăn cần tìm
        max_results (int): số lượng kết quả tối đa trả về

    Returns:
        List[dict]: danh sách dict, mỗi dict chứa {'dish_name': ...}
    """
    # Chuẩn hóa input
    category_lower = category.strip().lower()

    # Lọc DataFrame theo category
    df_filtered = df[df['category'].str.lower() == category_lower]

    # Lấy tối đa max_results món
    results = df_filtered.head(max_results)

    # Trả về list dict với key 'dish_name'
    return results[['dish_name']].to_dict(orient='records')
