# app/core/search_category.py
def search_dishes_by_category(df, category, max_results=5):
    category_clean = category.lower().strip()
    results = []
    for idx, row in df.iterrows():
        if str(row.get("category", "")).lower() == category_clean:
            results.append({"dish_name": row["dish_name"]})
    return results[:max_results]
