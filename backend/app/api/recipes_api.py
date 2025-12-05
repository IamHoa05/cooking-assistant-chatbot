# app/api/dish_search_service.py
from fastapi import APIRouter
from pydantic import BaseModel
import ast
import pandas as pd

from app.utils.faiss_handler import FAISSHandler
from app.utils.embedder import load_embedding_model, embed_texts
from app.core.llm_service import describe_dishes, smooth_instructions
from app.core.search_instructions import search_dish
from app.core.search_difficulty import get_dishes_by_difficulty
from app.core.search_time import search_dishes_by_cook_time
from app.core.search_servings import search_dishes_by_servings
from app.core.search_category import search_dishes_by_category
from app.core.search_engine import search_dishes, initialize_search_engine, search_by_ingredients
from app.utils.detect_intent_extract import detect_intent, extract_all_slots, format_output_by_intent

# Router FastAPI
router = APIRouter()

# Khởi tạo Search Engine
df, handler = initialize_search_engine()

# Embedding model BGE-M3
EMBED_MODEL_NAME = "BAAI/bge-m3"
embedding_model = load_embedding_model(EMBED_MODEL_NAME)

# Schema request
class TextQuery(BaseModel):
    text: str  # text input từ user

# API: Xử lý văn bản, detect intent và trích slot
@router.post("/process_text")
def process_text(query: TextQuery):
    """
    Xử lý input text từ người dùng:
    1. Detect intent
    2. Extract slots dựa vào intent
    3. Filter & search món ăn dựa vào các slot
    4. Nếu intent là suggest_dishes: LLM tạo mô tả
    5. Nếu intent là cooking_guide: trả hướng dẫn nấu
    """
    text = query.text
    print(f"==== DEBUG: Input user ====\n{text}\n")

    # Detect intent
    intent, score, _ = detect_intent(text)
    print(f"Detected intent: {intent}, score: {score}")

    # Extract slots dựa vào intent
    slots = extract_all_slots(text, intent=intent)
    print(f"Extracted slots: {slots}\n")

    # Xử lý intent 'suggest_dishes'
    if intent == "suggest_dishes":
        # Thứ tự filter: category → ingredient → time → difficulty → serving
        slot_order = ["category", "ingredient", "time", "difficulty", "serving"]
        candidates = None  # danh sách món khả thi ban đầu

        for slot in slot_order:
            value = slots.get(slot)
            if not value:
                continue  # bỏ qua nếu slot trống

            print(f"Processing slot: {slot}, value: {value}")

            if slot == "category":
                candidates = search_dishes_by_category(df, value, max_results=200)
                print(f"DEBUG: {len(candidates)} candidates after category filter")

            elif slot == "ingredient":
                ing_results = search_by_ingredients(value, df, handler, top_k=200)
                print(f"DEBUG: {len(ing_results)} candidates from ingredients search")

                if candidates is None:
                    candidates = ing_results
                else:
                    # giữ các món xuất hiện cả 2 filter
                    candidates = [
                        d for d in candidates
                        if d["dish_name"] in [r["dish_name"] for r in ing_results]
                    ]
                print(f"DEBUG: {len(candidates)} candidates after ingredient filter")

            elif slot == "time":
                time_val = value[0] if isinstance(value, list) else value
                # Nếu chưa có candidates → search từ df
                time_df = search_dishes_by_cook_time(df, time_val, max_results=200, tolerance=10)
                if candidates is None:
                    candidates = [{"dish_name": d["dish_name"]} for d in time_df]
                else:
                    candidates = [
                        d for d in candidates
                        if d["dish_name"] in [r["dish_name"] for r in time_df]
                    ]
                print(f"DEBUG: {len(candidates)} candidates after time filter")

            elif slot == "difficulty":
                diff_val = value
                diff_results = get_dishes_by_difficulty(df, difficulty=diff_val, top_k=200)
                if candidates is None:
                    candidates = diff_results
                else:
                    candidates = [
                        d for d in candidates
                        if d["dish_name"] in [r["dish_name"] for r in diff_results]
                    ]
                print(f"DEBUG: {len(candidates)} candidates after difficulty filter")

            elif slot == "serving":
                serving_val = value[0] if isinstance(value, list) else value
                serving_results = search_dishes_by_servings(df, handler, serving_val, top_k=200)
                if candidates is None:
                    candidates = serving_results
                else:
                    candidates = [
                        d for d in candidates
                        if d["dish_name"] in [r["dish_name"] for r in serving_results]
                    ]
                print(f"DEBUG: {len(candidates)} candidates after serving filter")

        # Lấy tên món ăn cuối cùng
        top_dishes = [d["dish_name"] for d in candidates][:10] if candidates else []

        if not top_dishes:
            return {
                "intent": intent,
                "top_dishes": [],
                "description": "Không tìm thấy món ăn phù hợp."
            }

        # LLM tạo mô tả món ăn
        description = describe_dishes(top_dishes, text)

        # Format output theo intent
        output = format_output_by_intent(intent, slots)
        output["top_dishes"] = top_dishes
        output["description"] = description

        print(f"==== DEBUG: Final top dishes ====\n{top_dishes}\n")
        return {"intent": intent, **output}

    # Xử lý intent 'cooking_guide'
    elif intent == "cooking_guide":
        dish_name = slots.get("dish_name")
        if not dish_name:
            return {
                "intent": intent,
                "dish_name": None,
                "error": "Không tìm thấy tên món để hướng dẫn."
            }

        # Search FAISS
        results = search_dish(dish_name, top_k=1)
        print(f"DEBUG: Results from search_dish: {results}")

        if not results:
            return {
                "intent": intent,
                "dish_name": dish_name,
                "error": f"Không tìm thấy món ăn phù hợp cho '{dish_name}'."
            }

        # Chọn món tốt nhất
        best = results[0]
        dish_name = best["dish_name"]
        metadata = best.get("metadata", {})

        # Parse ingredients, instructions, tips
        try:
            ingredients = ast.literal_eval(metadata.get("ingredients", [])) \
                if isinstance(metadata.get("ingredients"), str) else metadata.get("ingredients", [])
        except Exception:
            ingredients = []

        try:
            instructions = ast.literal_eval(metadata.get("instructions", [])) \
                if isinstance(metadata.get("instructions"), str) else metadata.get("instructions", [])
        except Exception:
            instructions = []

        try:
            tips = ast.literal_eval(metadata.get("tips", [])) \
                if isinstance(metadata.get("tips"), str) else metadata.get("tips", [])
        except Exception:
            tips = []

        image_link = metadata.get("image_link", "") or ""

        # Rewrite instructions bằng LLM
        steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

        print(f"DEBUG: ingredients={ingredients}")
        print(f"DEBUG: instructions={instructions}")
        print(f"DEBUG: steps_smooth={steps_smooth}")
        print(f"DEBUG: image_link={image_link}")

        return {
            "intent": intent,
            "dish_name": dish_name,
            "ingredients": ingredients,
            "instructions": instructions,
            "steps_smooth": steps_smooth,
            "tips": tips,
            "image_link": image_link
        }

    # Fallback: intent không xác định
    else:
        return {
            "intent": intent,
            "slots": slots,
            "error": "Không xác định được intent."
        }
