


# # app/api.py
# from xml.sax import handler
# from fastapi import APIRouter, FastAPI
# from pydantic import BaseModel
# from typing import List
# import os
# import pandas as pd
# import numpy as np
# import requests

# # utils
# from app.utils.faiss_handler import FAISSHandler
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# from langchain_groq import ChatGroq  # hoặc gemini API client
# from langchain_core.messages import SystemMessage, HumanMessage
# from app.core.search_engine import search_dishes

# # -------------------------
# # Load .env
# # -------------------------
# from dotenv import load_dotenv
# env_path = os.path.join(os.path.dirname(__file__), "../../.env")
# load_dotenv(env_path)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in .env")

# # -------------------------
# # FastAPI router
# # -------------------------
# router = APIRouter()

# # -------------------------
# # Load embeddings & FAISS
# # -------------------------
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

# # -------------------------
# # Input models
# # -------------------------
# class RecipeQuery(BaseModel):
#     ingredients: List[str]

# class SmoothQuery(BaseModel):
#     top_dishes: List[str]


# def search_recipes(query: RecipeQuery):
#     dishes = search_dishes(df, faiss_handler, query.ingredients)
#     return {"results": dishes}

# # -------------------------
# # API: smooth_recipes (Groq)
# # -------------------------

# def smooth_recipes(query: SmoothQuery):
#     if not query.top_dishes:
#         return {"text": "Không có món ăn nào để hiển thị."}

#     groq_chat = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.1-8b-instant"  # hoặc model Groq/Gemini bạn muốn
#     )

#     user_text = "\n".join([f"- {d}" for d in query.top_dishes])
    

#     messages = [
#     SystemMessage(
#         content=(
#             "Bạn là chuyên gia ẩm thực Việt Nam. "
#             "Hãy trả lời trực tiếp, cuốn hút và thân thiện, dựa trên nguyên liệu mà người dùng cung cấp. "
#             "Không liệt kê công thức, không chào hỏi dài dòng. "
#             "Tập trung vào trải nghiệm khi ăn, mùi vị, màu sắc và cảm giác. "
#             "Bạn có thể gợi ý món ăn hoặc hỏi người dùng muốn thử món nào."
#         )
#     ),
#     HumanMessage(
#         content=(
#             "Người dùng cung cấp nguyên liệu sau: {user_text}\n\n"
#             "Hãy viết đoạn chat như sau:\n"
#             "- Giới thiệu các món ăn phù hợp\n"
#             "- Mô tả cuốn hút, sinh động\n"
#             "- Có thể đưa ra gợi ý hoặc hỏi người dùng\n"
#             "Tối đa 3–5 câu, nhấn mạnh trải nghiệm, hương vị, màu sắc, cảm giác khi ăn."
#         )
#     )
# ]
#     response = groq_chat.generate([messages])  # lưu ý: phải bọc trong list!
#     output_text = response.generations[0][0].text

#     return {"text": output_text}

# # -------------------------
# # API: smart_recipes (search + smooth)
# # -------------------------
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):
#     # 1️⃣ Lấy top dishes từ hàm logic search_dishes
#     top_dishes = search_dishes(df, faiss_handler, query.ingredients)

#     # Debug: in ra terminal xem kết quả
#     print("DEBUG - top_dishes:", top_dishes)

#     # 2️⃣ Smooth bằng LLM/Groq
#     smooth_result = smooth_recipes(SmoothQuery(top_dishes=top_dishes))

#     return {
#         "top_dishes": top_dishes,           # <-- optional trả về frontend luôn
#         "description": smooth_result["text"]
#     }


# class DishName(BaseModel):
#     dish: str

# @router.post("/recipe_detail_llm")
# def recipe_detail_llm(query: DishName):
#     dish = query.dish.strip().lower()

#     # Tìm món ăn trong DataFrame
#     match = df[df["dish_name"].str.lower() == dish]

#     if match.empty:
#         return {
#             "error": f"Không tìm thấy công thức cho món '{query.dish}'."
#         }

#     row = match.iloc[0]

#     ingredients = row.get("ingredient_names", [])
#     steps = row.get("instructions", [])

#     # Chuẩn hóa steps nếu là string
#     if isinstance(steps, str):
#         steps = [s.strip() for s in steps.split("\n") if s.strip()]

#     # -------------------------
#     # LLM smoothing (Groq)
#     # -------------------------
#     groq_chat = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.1-8b-instant"
#     )

#     # Tạo prompt
#     ing_text = "\n".join([f"- {i}" for i in ingredients])
#     step_text = "\n".join([f"{idx+1}. {s}" for idx, s in enumerate(steps)])

#     messages = [
#         SystemMessage(
#             content=(
#                 "Bạn là đầu bếp chuyên nghiệp. "
#                 "Nhiệm vụ của bạn là viết lại hướng dẫn nấu ăn sao cho dễ hiểu, hấp dẫn và ngắn gọn. "
#                 "Không được thay đổi nguyên liệu hoặc tạo công thức mới."
#             )
#         ),
#         HumanMessage(
#             content=(
#                 f"Món ăn: {row['dish_name']}\n\n"
#                 f"Nguyên liệu:\n{ing_text}\n\n"
#                 f"Các bước nấu:\n{step_text}\n\n"
#                 "Hãy viết lại hướng dẫn nấu ăn theo dạng tự nhiên, hấp dẫn, tối đa 3–5 câu. "
#                 "Giải thích ngắn gọn các bước chính và tạo cảm giác ngon miệng khi đọc."
#             )
#         )
#     ]

#     response = groq_chat.generate([messages])
#     llm_text = response.generations[0][0].text.strip()

#     # Output
#     return {
#         "dish": row["dish_name"],
#         "ingredients": ingredients,
#         "steps_original": steps,
#         "steps_smooth": llm_text  # ✨ Hướng dẫn mượt từ LLM
#     }

# # app/api/recipes_api.py
# from fastapi import APIRouter
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pickle
# import os
# from difflib import get_close_matches

# # LLM/Groq
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage

# # Utils
# from app.core.search_engine import embed_texts, search_dishes  

# # -------------------------
# # Load environment
# # -------------------------
# from dotenv import load_dotenv
# env_path = os.path.join(os.path.dirname(__file__), "../../.env")
# load_dotenv(env_path)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in .env")

# # -------------------------
# # Router
# # -------------------------
# router = APIRouter()

# # -------------------------
# # Load embeddings + metadata
# # -------------------------
# EMB_PATH = "app/utils/data/recipes_embeddings.npy"
# META_PATH = "app/utils/data/metadata_lookup.pkl"

# if not os.path.exists(EMB_PATH):
#     raise FileNotFoundError(f"Embeddings file not found: {EMB_PATH}")
# if not os.path.exists(META_PATH):
#     raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

# embeddings = np.load(EMB_PATH)  # shape: (num_recipes, dim_total)
# with open(META_PATH, "rb") as f:
#     metadata = pickle.load(f)    # list of dicts [{dish_name, ingredient_names, instructions,...}, ...]

# print(f"✅ Loaded embeddings ({embeddings.shape}) and metadata ({len(metadata)} items)")

# # -------------------------
# # Input models
# # -------------------------
# class RecipeQuery(BaseModel):
#     ingredients: List[str]

# class DishName(BaseModel):
#     dish: str

# class SmoothQuery(BaseModel):
#     top_dishes: List[str]

# # -------------------------
# # Helper: LLM smooth description
# # -------------------------
# def smooth_recipes_groq(top_dishes: List[str]):
#     if not top_dishes:
#         return "Không có món ăn nào để hiển thị."

#     user_text = "\n".join([f"- {d}" for d in top_dishes])

#     groq_chat = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.1-8b-instant"
#     )

#     messages = [
#         SystemMessage(
#             content=(
#                 "Bạn là chuyên gia ẩm thực Việt Nam. "
#                 "Trả lời trực tiếp, cuốn hút và thân thiện, dựa trên nguyên liệu người dùng cung cấp. "
#                 "Không liệt kê công thức, không chào hỏi dài dòng. "
#                 "Tập trung trải nghiệm khi ăn, mùi vị, màu sắc và cảm giác."
#             )
#         ),
#         HumanMessage(
#             content=f"Người dùng cung cấp nguyên liệu:\n{user_text}\n\n"
#                     "Hãy viết đoạn chat 3–5 câu, giới thiệu món ăn, sinh động, cuốn hút."
#         )
#     ]

#     try:
#         response = groq_chat.generate([messages])
#         return response.generations[0][0].text.strip()
#     except Exception as e:
#         print("⚠️ Groq LLM error:", e)
#         return "Không thể tạo mô tả món ăn."

# # -------------------------
# # API: smart_recipes
# # -------------------------
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):
#     print("DEBUG: ingredients query:", query.ingredients)

#     if not query.ingredients:
#         return {"top_dishes": [], "description": "Vui lòng nhập nguyên liệu."}

#     # 1️⃣ Tạo embedding query từ nguyên liệu
#     query_vec = embed_texts(query.ingredients)  # shape: (1, dim_total)

#     # 2️⃣ FAISS search → trả index
#     try:
#         top_ids, scores = search_dishes(query_vec, embeddings, top_k=5)
#         print("DEBUG: top_ids:", top_ids)
#         print("DEBUG: scores:", scores)
#     except Exception as e:
#         print("⚠️ FAISS search error:", e)
#         return {"top_dishes": [], "description": "Có lỗi khi tìm kiếm món ăn."}

#     # 3️⃣ Lấy tên món ăn từ metadata
#     top_dishes = []
#     for idx in top_ids:
#         try:
#             top_dishes.append(metadata[idx]["dish_name"])
#         except IndexError:
#             print(f"⚠️ Index {idx} out of range for metadata")
    
#     print("DEBUG: top_dishes:", top_dishes)

#     # 4️⃣ Smooth bằng LLM/Groq
#     description = smooth_recipes_groq(top_dishes)

#     return {
#         "top_dishes": top_dishes,
#         "description": description
#     }

# # -------------------------
# # API: recipe_detail_llm
# # -------------------------
# @router.post("/recipe_detail_llm")
# def recipe_detail_llm(query: DishName):
#     dish_input = query.dish.strip()
#     print("DEBUG: user input dish:", repr(dish_input))

#     dish_names = [item["dish_name"].strip() for item in metadata]
#     closest = get_close_matches(dish_input, dish_names, n=1, cutoff=0.6)
#     print("DEBUG: closest match:", closest)

#     if not closest:
#         print("DEBUG: No match found in metadata")
#         return {"error": f"Không tìm thấy công thức cho món '{dish_input}'."}

#     # Lấy item metadata match
#     row = next(item for item in metadata if item["dish_name"].strip() == closest[0])
#     print("DEBUG: matched row dish_name:", row["dish_name"])

#     ingredients = row.get("ingredient_names", [])
#     instructions = row.get("instructions", [])
#     print("DEBUG: ingredients:", ingredients)
#     print("DEBUG: instructions:", instructions)

#     if isinstance(instructions, str):
#         instructions = [s.strip() for s in instructions.split("\n") if s.strip()]

#     # LLM smoothing
#     ing_text = "\n".join([f"- {i}" for i in ingredients])
#     step_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(instructions)])

#     try:
#         groq_chat = ChatGroq(
#             groq_api_key=GROQ_API_KEY,
#             model_name="llama-3.1-8b-instant"
#         )

#         messages = [
#             SystemMessage(
#                 content=(
#                     "Bạn là đầu bếp chuyên nghiệp. "
#                     "Viết lại hướng dẫn nấu ăn sao cho dễ hiểu, hấp dẫn, tối đa 3–5 câu. "
#                     "Không thay đổi nguyên liệu hoặc tạo công thức mới."
#                 )
#             ),
#             HumanMessage(
#                 content=f"Món ăn: {row['dish_name']}\n\nNguyên liệu:\n{ing_text}\n\n"
#                         f"Các bước nấu:\n{step_text}\n\nViết lại hướng dẫn nấu ăn ngắn gọn, hấp dẫn."
#             )
#         ]

#         response = groq_chat.generate([messages])
#         steps_smooth = response.generations[0][0].text.strip()
#         print("DEBUG: steps_smooth:", steps_smooth[:200])  # in 200 ký tự đầu
#     except Exception as e:
#         print("⚠️ LLM Groq error:", e)
#         steps_smooth = "Không thể tạo hướng dẫn mượt từ LLM."

#     return {
#         "dish": row["dish_name"],
#         "ingredients": ingredients,
#         "steps_original": instructions,
#         "steps_smooth": steps_smooth
#     }
# Refactored Pipeline 1 API (FastAPI)
# Combined: smart_recipes + recipe_detail_llm

# from fastapi import APIRouter, Depends
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pandas as pd
# import os
# from difflib import get_close_matches
# import ast

# # Models
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage

# # Utils
# from app.utils.faiss_handler import FAISSHandler
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# from app.core.search_instructions import search_dish

# from dotenv import load_dotenv

# # ================================
# # Load environment
# # ================================
# env_path = os.path.join(os.path.dirname(__file__), "../../.env")
# load_dotenv(env_path)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in .env")

# # ================================
# # Router
# # ================================
# router = APIRouter()

# # ================================
# # Load DataFrame + FAISS + Model
# # ================================
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

# # ================================
# # Input Schemas
# # ================================
# class RecipeQuery(BaseModel):
#     ingredients: List[str]

# class DishName(BaseModel):
#     dish: str

# class SmoothQuery(BaseModel):
#     top_dishes: List[str]

# # ================================
# # Helper: Groq smoothing
# # ================================
# def smooth_recipes(top_dishes: List[str]):
#     if not top_dishes:
#         return "Không có món ăn nào để hiển thị."

#     groq_chat = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.1-8b-instant"
#     )

#     dishes_text = "\n".join([f"- {d}" for d in top_dishes])

#     messages = [
#         SystemMessage(content=(
#             "Bạn là chuyên gia ẩm thực Việt Nam. "
#             "Viết đoạn giới thiệu ngắn (3–5 câu), mô tả sinh động về các món phù hợp."
#         )),
#         HumanMessage(content=(
#             f"Các món phù hợp với nguyên liệu bạn cung cấp:\n{dishes_text}\n"
#             "Hãy mô tả hấp dẫn, sinh động và tự nhiên."
#         ))
#     ]

#     try:
#         response = groq_chat.generate([messages])
#         return response.generations[0][0].text
#     except Exception:
#         return "Không thể tạo mô tả món ăn bằng LLM."

# # ================================
# # API: smart_recipes
# # ================================
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):

#     if not query.ingredients:
#         return {"top_dishes": [], "description": "Vui lòng nhập nguyên liệu."}

#     # Step 1: embed ingredient list
#     query_str = ", ".join(query.ingredients)
#     query_vec = embed_texts([query_str], embedding_model)[0]
#     query_vec = np.array(query_vec).astype("float32")

#     # Step 2: FAISS search
#     results = faiss_handler.search(
#         query_vector=query_vec,
#         column_key="names",
#         top_k=5
#     )

#     top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

#     # Step 3: Smooth via LLM
#     description = smooth_recipes(top_dishes)

#     return {
#         "top_dishes": top_dishes,
#         "description": description
#     }

# # ================================
# # API: recipe_detail_llm
# # ================================
# @router.post("/recipe_detail_llm")
# def recipe_detail_llm(query: DishName):
#     print("DEBUG: user input:", query.dish)

#     results = search_dish(query.dish, top_k=1)
#     print("DEBUG: search results:", results)

#     if not results:
#         print(f"❌ Không tìm thấy món ăn phù hợp cho '{query.dish}'")
#         return {"error": f"Không tìm thấy món ăn phù hợp cho '{query.dish}'."}

#     best = results[0]
#     dish_name = best["dish_name"]
#     score = best["score"]

#     # Parse ingredients & instructions nếu là string
#     ingredients_raw = best["metadata"]["ingredients"]
#     instructions_raw = best["metadata"]["instructions"]

#     try:
#         if isinstance(ingredients_raw, str):
#             ingredients = ast.literal_eval(ingredients_raw)
#         else:
#             ingredients = ingredients_raw
#     except Exception:
#         ingredients = []

#     try:
#         if isinstance(instructions_raw, str):
#             instructions = ast.literal_eval(instructions_raw)
#         else:
#             instructions = instructions_raw
#     except Exception:
#         instructions = []

#     print(f"✅ Tìm thấy món: {dish_name} (score: {score:.2f})")
#     print("Nguyên liệu:", ingredients)
#     print("Hướng dẫn gốc:", instructions)

#     # Build text cho LLM
#     ing_text = "\n".join([f"- {i}" for i in ingredients])
#     step_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(instructions)])

#     try:
#         groq_chat = ChatGroq(
#             groq_api_key=GROQ_API_KEY,
#             model_name="llama-3.1-8b-instant"
#         )

#         messages = [
#             SystemMessage(content=(
#                 "Bạn là đầu bếp chuyên nghiệp. "
#                 "Viết lại hướng dẫn nấu ăn sao cho dễ hiểu, hấp dẫn, tối đa 3–5 câu."
#             )),
#             HumanMessage(content=(
#                 f"Món ăn: {dish_name}\n"
#                 f"Nguyên liệu:\n{ing_text}\n\n"
#                 f"Các bước nấu:\n{step_text}\n\n"
#                 "Hãy viết lại hướng dẫn mượt và tự nhiên."
#             ))
#         ]

#         response = groq_chat.generate([messages])
#         steps_smooth = response.generations[0][0].text.strip()
#         print("✅ LLM steps_smooth:", steps_smooth[:200], "…")
#     except Exception as e:
#         steps_smooth = f"Không thể tạo hướng dẫn mượt bằng LLM: {e}"
#         print("⚠️ LLM error:", e)

#     return {
#         "dish_name": dish_name,
#         "score": score,
#         "ingredients": ingredients,
#         "instructions": instructions,
#         "steps_smooth": steps_smooth
#     }

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import ast

from app.core.search_instructions import search_dish
from app.core.llm_service import smooth_dishes, smooth_instructions, smooth_dishes_by_difficulty, smooth_by_cooking_time, smooth_by_servings, smooth_by_category
from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
from app.utils.faiss_handler import FAISSHandler
from app.core.search_difficulty import get_dishes_by_difficulty
from app.core.search_time import search_dishes_by_cook_time
from app.core.search_servings import search_dishes_by_servings
from app.core.search_category import search_dishes_by_category

router = APIRouter()

# Load DataFrame + FAISS + model
df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
embedding_columns = {
    "names": "ingredient_names_embedding",
    "quantities": "ingredient_quantities_embedding",
    "dish": "dish_name_embedding"
}
faiss_handler = FAISSHandler(
    df=df,
    embedding_columns=embedding_columns,
    index_dir="app/utils/faiss_indexes"
)
embedding_model = load_vietnamese_embedding_model()

# -----------------
# Schemas
# -----------------
class RecipeQuery(BaseModel):
    ingredients: List[str]

class DishName(BaseModel):
    dish: str

# -----------------
# API: smart_recipes
# -----------------
@router.post("/smart_recipes")
def smart_recipes(query: RecipeQuery):
    if not query.ingredients:
        return {"top_dishes": [], "description": "Vui lòng nhập nguyên liệu."}

    query_str = ", ".join(query.ingredients)
    query_vec = embed_texts([query_str], embedding_model)[0]
    query_vec = np.array(query_vec).astype("float32")

    results = faiss_handler.search(
        query_vector=query_vec,
        column_key="names",
        top_k=5
    )

    top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

    description = smooth_dishes(top_dishes)

    return {"top_dishes": top_dishes, "description": description}

# -----------------
# API: recipe_detail_llm
# -----------------
@router.post("/recipe_detail_llm")
def recipe_detail_llm(query: DishName):
    results = search_dish(query.dish, top_k=1)

    if not results:
        return {"error": f"Không tìm thấy món ăn phù hợp cho '{query.dish}'."}

    best = results[0]
    dish_name = best["dish_name"]
    metadata = best["metadata"]

    # parse ingredients & instructions
    try:
        ingredients = ast.literal_eval(metadata["ingredients"]) if isinstance(metadata["ingredients"], str) else metadata["ingredients"]
    except Exception:
        ingredients = []

    try:
        instructions = ast.literal_eval(metadata["instructions"]) if isinstance(metadata["instructions"], str) else metadata["instructions"]
    except Exception:
        instructions = []

    steps_smooth = smooth_instructions(dish_name, ingredients, instructions)

    return {
        "dish_name": dish_name,
        "ingredients": ingredients,
        "instructions": instructions,
        "steps_smooth": steps_smooth
    }


# -----------------
# API: recipes_by_difficulty
# -----------------
class DifficultyQuery(BaseModel):
    difficulty: str  # "easy" hoặc "medium"

@router.post("/recipes_by_difficulty")
def recipes_by_difficulty(query: DifficultyQuery):
    difficulty = query.difficulty.lower()
    if difficulty not in ["dễ", "trung bình"]:
        return {"top_dishes": [], "description": "Chỉ hỗ trợ độ khó: easy hoặc medium."}

    # 1️⃣ Tìm món ăn theo difficulty
    results = get_dishes_by_difficulty(df, difficulty=difficulty, top_k=5)
    top_dishes = [r.get("dish_name") for r in results if r.get("dish_name")]

    # 2️⃣ LLM smoothing
    description = smooth_dishes_by_difficulty(top_dishes, difficulty)

    return {
        "top_dishes": top_dishes,
        "description": description
    }


# -----------------
# API: recipes_by_time
# -----------------
class CookTimeQuery(BaseModel):
    minutes: int

router = APIRouter()

@router.post("/recipes_by_time")
def recipes_by_time(query: CookTimeQuery):
    target_time = query.minutes
    top_dishes_df = search_dishes_by_cook_time(df, target_time, max_results=5)
    top_dishes_names = top_dishes_df["dish_name"].tolist()

    description = smooth_by_cooking_time(top_dishes_names, target_time)

    return {
        "target_time": target_time,
        "top_dishes": top_dishes_names,
        "description": description
    }

# Input schema
class ServingsQuery(BaseModel):
    servings: int

@router.post("/recipes_by_servings")
def recipes_by_servings(query: ServingsQuery):
    target_servings = query.servings

    # 1️⃣ Tìm món ăn gần nhất
    top_dishes_meta = search_dishes_by_servings(df, faiss_handler, target_servings, top_k=5)
    top_dishes_names = [d["dish_name"] for d in top_dishes_meta]

    # 2️⃣ Smooth bằng LLM/Groq
    description = smooth_by_servings(top_dishes_names)

    return {
        "target_servings": target_servings,
        "top_dishes": top_dishes_names,
        "description": description
    }

class CategoryQuery(BaseModel):
    category: str

@router.post("/recipes_by_category")
def recipes_by_category(query: CategoryQuery):
    # 1️⃣ Tìm món
    top_dishes_meta = search_dishes_by_category(df, query.category, max_results=5)
    top_dishes_names = [d["dish_name"] for d in top_dishes_meta]

    # 2️⃣ Smooth bằng LLM
    description = smooth_by_category(top_dishes_names, category=query.category)

    return {
        "category": query.category,
        "top_dishes": top_dishes_names,
        "description": description
    }