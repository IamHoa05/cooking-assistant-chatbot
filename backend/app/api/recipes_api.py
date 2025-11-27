# # from fastapi import APIRouter
# # from pydantic import BaseModel
# # from typing import List
# # import numpy as np
# # import pandas as pd
# # import ast

# # from app.core.search_instructions import search_dish
# # from app.core.llm_service import smooth_dishes, smooth_instructions, smooth_dishes_by_difficulty, smooth_by_cooking_time, smooth_by_servings, smooth_by_category
# # from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# # from app.utils.faiss_handler import FAISSHandler
# # from app.core.search_difficulty import get_dishes_by_difficulty
# # from app.core.search_time import search_dishes_by_cook_time
# # from app.core.search_servings import search_dishes_by_servings
# # from app.core.search_category import search_dishes_by_category

# # router = APIRouter()

# # # Load DataFrame + FAISS + model
# # df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
# # embedding_columns = {
# #     "names": "ingredient_names_embedding",
# #     "quantities": "ingredient_quantities_embedding",
# #     "dish": "dish_name_embedding"
# # }
# # faiss_handler = FAISSHandler(
# #     df=df,
# #     embedding_columns=embedding_columns,
# #     index_dir="app/utils/faiss_indexes"
# # )
# # embedding_model = load_vietnamese_embedding_model()

# # # -----------------
# # # Schemas
# # # -----------------
# # class RecipeQuery(BaseModel):
# #     ingredients: List[str]

# # class DishName(BaseModel):
# #     dish: str

# # # -----------------
# # # API: smart_recipes
# # # -----------------
# # @router.post("/smart_recipes")
# # def smart_recipes(query: RecipeQuery):
# #     if not query.ingredients:
# #         return {"top_dishes": [], "description": "Vui lòng nhập nguyên liệu."}

# #     query_str = ", ".join(query.ingredients)
# #     query_vec = embed_texts([query_str], embedding_model)[0]
# #     query_vec = np.array(query_vec).astype("float32")

# #     results = faiss_handler.search(
# #         query_vector=query_vec,
# #         column_key="names",
# #         top_k=5
# #     )

# #     top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

# #     description = smooth_dishes(top_dishes)

# #     return {"top_dishes": top_dishes, "description": description}

# # # -----------------
# # # API: recipe_detail_llm
# # # -----------------
# # @router.post("/recipe_detail_llm")
# # def recipe_detail_llm(query: DishName):
# #     results = search_dish(query.dish, top_k=1)

# #     if not results:
# #         return {"error": f"Không tìm thấy món ăn phù hợp cho '{query.dish}'."}

# #     best = results[0]
# #     dish_name = best["dish_name"]
# #     metadata = best["metadata"]

# #     # parse ingredients & instructions
# #     try:
# #         ingredients = ast.literal_eval(metadata["ingredients"]) if isinstance(metadata["ingredients"], str) else metadata["ingredients"]
# #     except Exception:
# #         ingredients = []

# #     try:
# #         instructions = ast.literal_eval(metadata["instructions"]) if isinstance(metadata["instructions"], str) else metadata["instructions"]
# #     except Exception:
# #         instructions = []

# #     steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

# #     return {
# #         "dish_name": dish_name,
# #         "ingredients": ingredients,
# #         "instructions": instructions,
# #         "steps_smooth": steps_smooth
# #     }


# # # # -----------------
# # # # API: recipes_by_difficulty
# # # # -----------------
# # # class DifficultyQuery(BaseModel):
# # #     difficulty: str  # "easy" hoặc "medium"

# # # @router.post("/recipes_by_difficulty")
# # # def recipes_by_difficulty(query: DifficultyQuery):
# # #     difficulty = query.difficulty.lower()
# # #     if difficulty not in ["dễ", "trung bình"]:
# # #         return {"top_dishes": [], "description": "Chỉ hỗ trợ độ khó: easy hoặc medium."}

# # #     # 1️⃣ Tìm món ăn theo difficulty
# # #     results = get_dishes_by_difficulty(df, difficulty=difficulty, top_k=5)
# # #     top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

# # #     # 2️⃣ LLM smoothing
# # #     description = smooth_dishes_by_difficulty(top_dishes, difficulty)

# # #     return {
# # #         "top_dishes": top_dishes,
# # #         "description": description
# # #     }


# # # # -----------------
# # # # API: recipes_by_time
# # # # -----------------
# # # class CookTimeQuery(BaseModel):
# # #     minutes: int

# # # @router.post("/recipes_by_time")
# # # def recipes_by_time(query: CookTimeQuery):
# # #     target_time = query.minutes
# # #     top_dishes_df = search_dishes_by_cook_time(df, target_time, max_results=5)
# # #     top_dishes_names = top_dishes_df["dish_name"].tolist()

# # #     description = smooth_by_cooking_time(top_dishes_names, target_time)

# # #     return {
# # #         "target_time": target_time,
# # #         "top_dishes": top_dishes_names,
# # #         "description": description
# # #     }

# # # # Input schema
# # # class ServingsQuery(BaseModel):
# # #     servings: int

# # # @router.post("/recipes_by_servings")
# # # def recipes_by_servings(query: ServingsQuery):
# # #     target_servings = query.servings

# # #     # 1️⃣ Tìm món ăn gần nhất
# # #     top_dishes_meta = search_dishes_by_servings(df, faiss_handler, target_servings, top_k=5)
# # #     top_dishes_names = [d["dish_name"] for d in top_dishes_meta]

# # #     # 2️⃣ Smooth bằng LLM/Groq
# # #     description = smooth_by_servings(top_dishes_names)

# # #     return {
# # #         "target_servings": target_servings,
# # #         "top_dishes": top_dishes_names,
# # #         "description": description
# # #     }

# # # class CategoryQuery(BaseModel):
# # #     category: str

# # # @router.post("/recipes_by_category")
# # # def recipes_by_category(query: CategoryQuery):
# # #     # 1️⃣ Tìm món
# # #     top_dishes_meta = search_dishes_by_category(df, query.category, max_results=5)
# # #     top_dishes_names = [d["dish_name"] for d in top_dishes_meta]

# # #     # 2️⃣ Smooth bằng LLM
# # #     description = smooth_by_category(top_dishes_names, category=query.category)

# # #     return {
# # #         "category": query.category,
# # #         "top_dishes": top_dishes_names,
# # #         "description": description
# # #     }

# from fastapi import APIRouter
# from pydantic import BaseModel
# from typing import List, Optional
# import numpy as np
# import pandas as pd
# import ast

# from app.core.search_instructions import search_dish
# from app.core.llm_service import (
#     smooth_dishes,
#     smooth_instructions,
#     smooth_dishes_by_difficulty,
#     smooth_by_cooking_time,
#     smooth_by_servings,
#     smooth_by_category
# )
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# from app.utils.faiss_handler import FAISSHandler
# from app.core.search_difficulty import get_dishes_by_difficulty
# from app.core.search_time import search_dishes_by_cook_time
# from app.core.search_servings import search_dishes_by_servings
# from app.core.search_category import search_dishes_by_category

# router = APIRouter()

# # -----------------
# # Load DataFrame + FAISS + embedding model
# # -----------------
# df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
# embedding_columns = {
#     "names": "ingredient_names_embedding",
#     "quantities": "ingredient_quantities_embedding",
#     "dish": "dish_name_embedding"
# }
# faiss_handler = FAISSHandler(
#     df=df,
#     embedding_columns=embedding_columns,
#     index_dir="app/utils/faiss_indexes"
# )
# embedding_model = load_vietnamese_embedding_model()

# # -----------------
# # Schemas
# # -----------------
# class RecipeQuery(BaseModel):
#     ingredients: List[str]
#     difficulty: Optional[str] = None
#     time: Optional[int] = None
#     servings: Optional[int] = None
#     category: Optional[str] = None

# class DishName(BaseModel):
#     dish: str

# # -----------------
# # API: smart_recipes (Ingredient Suggestion + LLM smoothing)
# # -----------------
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):
#     # 1️⃣ FAISS search bằng nguyên liệu
#     top_dishes = []
#     if query.ingredients:
#         query_str = ", ".join(query.ingredients)
#         query_vec = embed_texts([query_str], embedding_model)[0]
#         query_vec = np.array(query_vec).astype("float32")
#         results = faiss_handler.search(
#             query_vector=query_vec,
#             column_key="names",
#             top_k=20
#         )
#         top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

#     # 2️⃣ Lọc theo các slot khác (difficulty, time, servings, category)
#     if query.difficulty:
#         top_dishes = smooth_dishes_by_difficulty(top_dishes, query.difficulty)

#     if query.time:
#         top_dishes = smooth_by_cooking_time(top_dishes, query.time)

#     if query.servings:
#         top_dishes = smooth_by_servings(top_dishes, query.servings)

#     if query.category:
#         top_dishes = smooth_by_category(top_dishes, query.category)

#     # 3️⃣ LLM smoothing description
#     description = smooth_dishes(top_dishes)

#     return {
#         "top_dishes": top_dishes,
#         "description": description
#     }

# # -----------------
# # API: recipe_detail_llm (Recipe detail + LLM smoothing)
# # -----------------
# @router.post("/recipe_detail_llm")
# def recipe_detail_llm(query: DishName):
#     results = search_dish(query.dish, top_k=1)

#     if not results:
#         return {"error": f"Không tìm thấy món ăn phù hợp cho '{query.dish}'."}

#     best = results[0]
#     dish_name = best["dish_name"]
#     metadata = best.get("metadata", {})

#     # parse ingredients & instructions
#     try:
#         ingredients = ast.literal_eval(metadata.get("ingredients", [])) if isinstance(metadata.get("ingredients"), str) else metadata.get("ingredients", [])
#     except Exception:
#         ingredients = []

#     try:
#         instructions = ast.literal_eval(metadata.get("instructions", [])) if isinstance(metadata.get("instructions"), str) else metadata.get("instructions", [])
#     except Exception:
#         instructions = []

#     # LLM smoothing steps
#     steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

#     return {
#         "dish_name": dish_name,
#         "ingredients": ingredients,
#         "instructions": instructions,
#         "steps_smooth": steps_smooth
#     }


# from fastapi import APIRouter
# from pydantic import BaseModel
# from typing import List, Optional
# import numpy as np
# import pandas as pd
# import ast

# from app.core.search_instructions import search_dish
# from app.core.llm_service import describe_dishes, smooth_instructions
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# from app.utils.faiss_handler import FAISSHandler

# router = APIRouter()

# # -----------------
# # Load DataFrame + FAISS + embedding model
# # -----------------
# df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
# embedding_columns = {
#     "names": "ingredient_names_embedding",
#     "dish": "dish_name_embedding"
# }
# faiss_handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="app/utils/faiss_indexes")
# embedding_model = load_vietnamese_embedding_model()

# # -----------------
# # Schemas
# # -----------------
# class RecipeQuery(BaseModel):
#     ingredients: Optional[List[str]] = None
#     difficulty: Optional[str] = None
#     time: Optional[int] = None
#     servings: Optional[int] = None
#     category: Optional[str] = None

# class DishName(BaseModel):
#     dish: str

# # -----------------
# # API: smart_recipes (2 intent: search + description)
# # -----------------
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):
#     """
#     Flow:
#     1. Tìm món theo nguyên liệu (FAISS)
#     2. Lọc tuần tự theo các slot còn lại nếu có
#     3. Lấy ra danh sách món cuối cùng và dùng LLM tạo mô tả
#     """
#     top_dishes = []

#     # 1️⃣ Tìm theo nguyên liệu
#     if query.ingredients:
#         query_str = ", ".join(query.ingredients)
#         query_vec = embed_texts([query_str], embedding_model)[0]
#         query_vec = np.array(query_vec).astype("float32")
#         results = faiss_handler.search(query_vector=query_vec, column_key="names", top_k=50)
#         top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

#     # Nếu không có món nào, trả về trống
#     if not top_dishes:
#         return {"top_dishes": [], "description": "Không tìm thấy món ăn phù hợp."}

#     # 2️⃣ Lọc tuần tự theo các slot
#     # Lưu ý: search functions trả về list món còn giữ tên món
#     if query.difficulty:
#         from app.core.search_difficulty import get_dishes_by_difficulty
#         top_dishes = [r["dish_name"] for r in get_dishes_by_difficulty(df, difficulty=query.difficulty, top_k=50) if r["dish_name"] in top_dishes]

#     if query.time:
#         from app.core.search_time import search_dishes_by_cook_time
#         time_df = search_dishes_by_cook_time(df, query.time, max_results=50)
#         top_dishes = [d for d in time_df["dish_name"].tolist() if d in top_dishes]

#     if query.servings:
#         from app.core.search_servings import search_dishes_by_servings
#         servings_meta = search_dishes_by_servings(df, faiss_handler, query.servings, top_k=50)
#         top_dishes = [d["dish_name"] for d in servings_meta if d["dish_name"] in top_dishes]

#     if query.category:
#         from app.core.search_category import search_dishes_by_category
#         cat_meta = search_dishes_by_category(df, query.category, max_results=50)
#         top_dishes = [d["dish_name"] for d in cat_meta if d["dish_name"] in top_dishes]

#     # 3️⃣ LLM tạo mô tả hấp dẫn
#     description = describe_dishes(top_dishes)

#     return {
#         "top_dishes": top_dishes,
#         "description": description
#     }

# # -----------------
# # API: recipe_detail_llm (Detail + LLM smoothing)
# # -----------------
# @router.post("/recipe_detail_llm")
# def recipe_detail_llm(query: DishName):
#     results = search_dish(query.dish, top_k=1)
#     if not results:
#         return {"error": f"Không tìm thấy món ăn phù hợp cho '{query.dish}'."}

#     best = results[0]
#     dish_name = best["dish_name"]
#     metadata = best.get("metadata", {})

#     # parse ingredients & instructions
#     try:
#         ingredients = ast.literal_eval(metadata.get("ingredients", [])) if isinstance(metadata.get("ingredients"), str) else metadata.get("ingredients", [])
#     except Exception:
#         ingredients = []

#     try:
#         instructions = ast.literal_eval(metadata.get("instructions", [])) if isinstance(metadata.get("instructions"), str) else metadata.get("instructions", [])
#     except Exception:
#         instructions = []

#     # LLM smoothing
#     steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

#     return {
#         "dish_name": dish_name,
#         "ingredients": ingredients,
#         "instructions": instructions,
#         "steps_smooth": steps_smooth
#     }


from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
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
        description = describe_dishes(top_dishes)

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
