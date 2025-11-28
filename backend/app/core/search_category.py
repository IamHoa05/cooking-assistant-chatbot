# app/core/search_category.py
import pandas as pd

def search_dishes_by_category(df: pd.DataFrame, category: str, max_results: int = 5):
    """
    Tìm món ăn theo thể loại trực tiếp trên DataFrame (không dùng metadata).
    
    Args:
        df: DataFrame chứa cột 'dish_name' và 'category'
        category: thể loại món ăn
        max_results: số món trả về tối đa
    """
    category_lower = category.strip().lower()
    df_filtered = df[df['category'].str.lower() == category_lower]

    # Lấy top max_results
    results = df_filtered.head(max_results)

    # Trả về list dict với dish_name
    return results[['dish_name']].to_dict(orient='records')
