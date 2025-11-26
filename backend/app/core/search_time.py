# app/core/search_time.py
import pandas as pd

def search_dishes_by_cook_time(df: pd.DataFrame, target_time: int, max_results: int = 5):
    # Loại bỏ các món không có cook_time
    df_time = df.dropna(subset=["cooking_time"]).copy()
    # Tính khoảng cách với target_time
    df_time["distance"] = (df_time["cooking_time"] - target_time).abs()
    # Sắp xếp theo khoảng cách nhỏ nhất
    df_time = df_time.sort_values("distance")
    # Lấy top max_results
    return df_time.head(max_results)