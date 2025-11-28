from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import ast
import pandas as pd

from app.utils.faiss_handler import FAISSHandler
from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
from app.core.llm_service import describe_dishes, smooth_instructions
from app.core.search_instructions import search_dish
from app.core.search_difficulty import get_dishes_by_difficulty
from app.core.search_time import search_dishes_by_cook_time
from app.core.search_servings import search_dishes_by_servings
from app.core.search_category import search_dishes_by_category

# Import các hàm detect intent & extract slot từ file logic trước
from app.utils.detect_intent_extract import detect_intent, extract_all_slots, format_output_by_intent

router = APIRouter()

# -----------------
# Load DataFrame + FAISS + embedding model
# -----------------
df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
embedding_columns = {
    "names": "ingredient_names_embedding",
    "dish": "dish_name_embedding"
}
faiss_handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="app/utils/faiss_indexes")
embedding_model = load_vietnamese_embedding_model()

# -----------------
# Schema
# -----------------
class TextQuery(BaseModel):
    text: str

# -----------------
# API duy nhất
# -----------------
@router.post("/process_text")
def process_text(query: TextQuery):
    text = query.text

    # 1️⃣ Detect intent
    intent, score, _ = detect_intent(text)

    # 2️⃣ Extract all slots
    slots = extract_all_slots(text)

    # 3️⃣ Xử lý theo intent
    if intent == "suggest_dishes":
        # Start from all dishes
        top_dishes = []

        # 3.1 Tìm theo nguyên liệu
        if slots.get("ingredient"):
            query_str = ", ".join(slots["ingredient"])
            query_vec = embed_texts([query_str], embedding_model)[0]
            query_vec = np.array(query_vec).astype("float32")
            results = faiss_handler.search(query_vector=query_vec, column_key="names", top_k=50)
            top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

        # Nếu không có món nào
        if not top_dishes:
            return {"intent": intent, "top_dishes": [], "description": "Không tìm thấy món ăn phù hợp."}

        # 3.2 Lọc tuần tự theo các slot khác
        if slots.get("difficulty"):
            top_dishes = [r["dish_name"] for r in get_dishes_by_difficulty(df, difficulty=slots["difficulty"], top_k=50) if r["dish_name"] in top_dishes]

        if slots.get("time"):
            time_df = search_dishes_by_cook_time(df, slots["time"][0], max_results=50)
            top_dishes = [d for d in time_df["dish_name"].tolist() if d in top_dishes]

        if slots.get("serving"):
            servings_meta = search_dishes_by_servings(df, faiss_handler, slots["serving"][0], top_k=50)
            top_dishes = [d["dish_name"] for d in servings_meta if d["dish_name"] in top_dishes]

        if slots.get("category"):
            cat_meta = search_dishes_by_category(df, slots["category"], max_results=50)
            top_dishes = [d["dish_name"] for d in cat_meta if d["dish_name"] in top_dishes]

        # 3.3 LLM tạo mô tả
        description = describe_dishes(top_dishes,text)

        # 3.4 Format output
        output = format_output_by_intent(intent, slots)
        output["top_dishes"] = top_dishes
        output["description"] = description
        return {"intent": intent, **output}

    elif intent == "cooking_guide":
        # 3.1 Tìm theo dish_name
        dish_name = slots.get("dish_name")
        if not dish_name:
            return {"intent": intent, "dish_name": None, "error": "Không tìm thấy tên món để hướng dẫn."}

        results = search_dish(dish_name, top_k=1)
        if not results:
            return {"intent": intent, "dish_name": dish_name, "error": f"Không tìm thấy món ăn phù hợp cho '{dish_name}'."}

        best = results[0]
        dish_name = best["dish_name"]
        metadata = best.get("metadata", {})

        try:
            ingredients = ast.literal_eval(metadata.get("ingredients", [])) if isinstance(metadata.get("ingredients"), str) else metadata.get("ingredients", [])
        except Exception:
            ingredients = []

        try:
            instructions = ast.literal_eval(metadata.get("instructions", [])) if isinstance(metadata.get("instructions"), str) else metadata.get("instructions", [])
        except Exception:
            instructions = []

        # LLM smoothing
        steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

        return {
            "intent": intent,
            "dish_name": dish_name,
            "ingredients": ingredients,
            "instructions": instructions,
            "steps_smooth": steps_smooth
        }
    else:
        # fallback
        return {"intent": intent, "slots": slots, "error": "Không xác định được intent."}
