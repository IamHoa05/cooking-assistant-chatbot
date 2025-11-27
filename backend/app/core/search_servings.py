# app/core/search_servings.py
def search_dishes_by_servings(df, servings, top_k=5):
    results = []
    for idx, row in df.iterrows():
        try:
            row_servings = int(row.get("servings", 0))
        except:
            row_servings = 0
        if row_servings == servings:
            results.append({"dish_name": row["dish_name"]})
    return results[:top_k]
