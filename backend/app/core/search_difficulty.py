# app/core/search_difficulty.py
def get_dishes_by_difficulty(df, difficulty, top_k=5):
    difficulty_clean = difficulty.lower().strip()
    results = []
    for idx, row in df.iterrows():
        if str(row.get("difficulty", "")).lower() == difficulty_clean:
            results.append({"dish_name": row["dish_name"]})
    return results[:top_k]
