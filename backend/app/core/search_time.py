# app/core/search_time.py

import pandas as pd

def search_dishes_by_cook_time(df: pd.DataFrame, target_time: int, max_results: int = 5, tolerance: int = 10):
    """
    Tìm món ăn gần đúng với thời gian nấu mục tiêu, cho phép ±tolerance phút.

    Args:
        df (pd.DataFrame): DataFrame chứa recipes, cần có cột 'dish_name' và 'cooking_time'
        target_time (int): thời gian nấu mục tiêu (phút)
        max_results (int): số món trả về tối đa
        tolerance (int): khoảng thời gian cho phép ±tolerance phút

    Returns:
        List[dict]: danh sách món ăn gần đúng với thời gian, mỗi món gồm:
            - dish_name
            - cooking_time
            - score: 1 / (1 + |cooking_time - target_time|) → càng gần càng cao
    """
    # Loại bỏ món không có cooking_time
    df_time = df.dropna(subset=["cooking_time"]).copy()

    # Chỉ giữ món trong khoảng target_time ± tolerance
    df_time = df_time[
        df_time["cooking_time"].between(target_time - tolerance, target_time + tolerance)
    ].copy()

    # Tính khoảng cách tuyệt đối
    df_time["distance"] = (df_time["cooking_time"] - target_time).abs()

    # Tính score: càng gần target_time càng cao
    df_time["score"] = 1 / (1 + df_time["distance"])

    # Sắp xếp theo score giảm dần
    df_time = df_time.sort_values("score", ascending=False)

    # Lấy top max_results
    top_rows = df_time.head(max_results)

    # Trả về danh sách dict
    results = [
        {
            "dish_name": row["dish_name"],
            "cooking_time": int(row["cooking_time"]),
            "score": float(row["score"])
        }
        for _, row in top_rows.iterrows()
    ]

    return results
