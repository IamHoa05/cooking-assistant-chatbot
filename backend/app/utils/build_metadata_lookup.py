import os
import pickle
import pandas as pd
import numpy as np


def save_metadata_lookup(df_emb_vi: pd.DataFrame, output_path: str = "./data/metadata_lookup.pkl"):
    """
    Build and save metadata lookup file without embeddings.
    """

    # 🔹 Ensure there is a unique ID per record
    if "id" not in df_emb_vi.columns:
        df_emb_vi["id"] = df_emb_vi.index

    # 🔹 Fields for display / lookup
    display_fields = [
    "id", "dish_name", "ingredient_names", "ingredient_quantities", "instructions",
    "cooking_time", "servings", "difficulty", "usage", "tips", "image_link"
]

    # 🔹 Check missing columns
    available_fields = [col for col in display_fields if col in df_emb_vi.columns]
    missing_fields = set(display_fields) - set(available_fields)

    if missing_fields:
        print(f"⚠️ Missing columns in DataFrame: {missing_fields}")

    # 🔹 Extract metadata rows
    df_meta = df_emb_vi[available_fields].fillna("").copy()

    # 🔹 Convert numpy types → Python types (for safe JSON serialization later)
    def convert_type(value):
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if value is None or pd.isna(value):
            return ""
        return value

    metadata = []
    for _, row in df_meta.iterrows():
        cleaned = {k: convert_type(v) for k, v in row.items()}
        metadata.append(cleaned)

    # 🔹 Ensure folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 🔹 Save file
    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved metadata ({len(metadata)} items) → {output_path}")
    return metadata


# ========== RUNNING ==========
if __name__ == "__main__":
    df_emb_vi = pd.read_pickle("./data/recipes_embeddings.pkl")
    save_metadata_lookup(df_emb_vi)
