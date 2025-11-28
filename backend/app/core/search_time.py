# app/core/search_time.py
def search_dishes_by_cook_time(df, cook_time, max_results=5):
    results = []
    for idx, row in df.iterrows():
        try:
            row_time = int(row.get("cooking_time", 0))
        except:
            row_time = 0
        if row_time <= cook_time:  # tìm món nấu <= cook_time
            results.append({"dish_name": row["dish_name"]})
    return results[:max_results]
