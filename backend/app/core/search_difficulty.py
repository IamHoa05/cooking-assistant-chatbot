# app/core/search_difficulty.py

import pandas as pd

def get_dishes_by_difficulty(df: pd.DataFrame, difficulty: str, top_k: int = 5):
    """
    Tìm món ăn theo độ khó.

    Args:
        df: DataFrame chứa recipes, cần có cột 'dish_name' và 'difficulty'
        difficulty: 'easy' hoặc 'medium'
        top_k: số món trả về

    Returns:
        List[dict]: mỗi dict chứa ít nhất 'dish_name', 'difficulty'
    """
    difficulty = difficulty.lower()
    df_filtered = df[df["difficulty"].str.lower() == difficulty]
    top_dishes = df_filtered.head(top_k).to_dict(orient="records")
    return top_dishes
