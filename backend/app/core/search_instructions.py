# app/core/search_instructions.py
def search_dish(df, dish_name, top_k=1):
    name_clean = dish_name.lower().strip()
    results = []
    for idx, row in df.iterrows():
        if name_clean in str(row.get("dish_name", "")).lower():
            results.append({"dish_name": row["dish_name"]})
    return results[:top_k]
