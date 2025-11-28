# from fastapi import APIRouter, FastAPI
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from typing import List
# import pandas as pd
# import numpy as np
# from app.utils.faiss_handler import FAISSHandler
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
# import os
# import requests

# router = APIRouter()


# # Load data & FAISS
# df = pd.read_pickle("app/utils/data/recipes_embeddings.pkl")
# embedding_columns = {
#     "names": "ingredient_names_embedding",
#     "quantities": "ingredient_quantities_embedding",
#     "dish": "dish_name_embedding"
# }
# faiss_handler = FAISSHandler(df=df, embedding_columns=embedding_columns, index_dir="app/utils/data/faiss_indexes")
# embedding_model = load_vietnamese_embedding_model()

# class RecipeQuery(BaseModel):
#     ingredients: List[str]

# @router.post("/search_recipes")
# def search_recipes(query: RecipeQuery):
#     if not query.ingredients:
#         return {"results": []}

#     query_text = ", ".join(query.ingredients)
#     query_vector = embed_texts([query_text], embedding_model, text_type="query")[0]
#     query_vector = np.array(query_vector).astype("float32")

#     results = faiss_handler.search(query_vector=query_vector, column_key="names", top_k=3)

#     output = []
#     for r in results:
#         output.append({
#             "dish_name": r.get("dish_name")
#             # "ingredients": r.get("ingredient_names", []),
#             # "distance": float(r["_distance"])
#         })

#     return {"results": output}

# app/api/recipes.py
# from fastapi import APIRouter, FastAPI
# from pydantic import BaseModel
# from typing import List
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import numpy as np
# import requests

# # utils
# from app.utils.faiss_handler import FAISSHandler
# from app.utils.embedder import load_vietnamese_embedding_model, embed_texts

# # -------------------------
# # Load .env & Groq API key
# # -------------------------
# # load .env nằm cùng cấp với api.py
# env_path = os.path.join(os.path.dirname(__file__), ".env")
# load_dotenv(env_path)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found in .env")

# GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateMessage"

# HEADERS = {
#     "Authorization": f"Bearer {GEMINI_API_KEY}",
#     "Content-Type": "application/json"
# }


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
#     index_dir="app/utils/data/faiss_indexes"
# )

# embedding_model = load_vietnamese_embedding_model()

# # -------------------------
# # Input models
# # -------------------------
# class RecipeQuery(BaseModel):
#     ingredients: List[str]

# class SmoothQuery(BaseModel):
#     top_dishes: List[str]

# # -------------------------
# # API: search_recipes
# # -------------------------
# @router.post("/search_recipes")
# def search_recipes(query: RecipeQuery):
#     if not query.ingredients:
#         return {"results": []}

#     query_text = ", ".join(query.ingredients)
#     query_vector = embed_texts([query_text], embedding_model, text_type="query")[0]
#     query_vector = np.array(query_vector).astype("float32")

#     results = faiss_handler.search(
#         query_vector=query_vector,
#         column_key="names",
#         top_k=3
#     )

#     output = []
#     for r in results:
#         output.append({
#             "dish_name": r.get("dish_name")
#             # Có thể thêm "ingredients" hoặc "_distance" nếu muốn
#         })

#     return {"results": output}

# # -------------------------
# # API: smooth_recipes (Groq)
# # -------------------------
# @router.post("/smooth_recipes")
# def smooth_recipes(query: SmoothQuery):
#     if not query.top_dishes:
#         return {"text": "Không có món ăn nào để hiển thị."}

#     prompt = (
#         "Bạn là chuyên gia ẩm thực Việt Nam. Hãy viết lại danh sách các món ăn "
#         "dưới đây một cách mượt mà, ngắn gọn, dễ đọc, hấp dẫn. "
#         "Giữ đúng tên món ăn.\n\n"
#         "Danh sách món ăn:\n" +
#         "\n".join([f"- {d}" for d in query.top_dishes]) +
#         "\n\nHãy viết tối đa 3–5 dòng, súc tích và hấp dẫn."
#     )

#     payload = {
#         "prompt": prompt,
#         "temperature": 0.7,
#         "maxOutputTokens": 100
#     }

#     response = requests.post(GEMINI_API_URL, headers=HEADERS, json=payload)
#     if response.status_code != 200:
#         return {"text": f"Error: {response.status_code}, {response.text}"}

#     try:
#         output_text = response.json().get("candidates", [{}])[0].get("content", "")
#     except Exception:
#         output_text = "Model trả về kết quả không đúng định dạng."

#     return {"text": output_text}

# # -------------------------
# # API: smart_recipes (search + smooth)
# # -------------------------
# @router.post("/smart_recipes")
# def smart_recipes(query: RecipeQuery):
#     # 1. Lấy top dishes từ search_recipes
#     search_result = search_recipes(query)
#     top_dishes = [d["dish_name"] for d in search_result["results"]]

#     # 2. Smooth bằng LLM
#     smooth_result = smooth_recipes(SmoothQuery(top_dishes=top_dishes))

#     return {
#         "top_dishes": top_dishes,
#         "description": smooth_result["text"]
#     }

# # # -------------------------
# # # FastAPI app (nếu chạy standalone)
# # # -------------------------
# # app = FastAPI(title="Recipes API")
# # app.include_router(router)


# app/api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import pandas as pd
import numpy as np

# utils
from app.utils.new_faiss_handler import FAISSHandler
from app.utils.new_embedder import load_embedding_model, embed_texts
# Tạm comment Groq nếu chưa cần
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage
from app.core.new_search_engine import search_dishes, initialize_search_engine

# -------------------------
# Load .env
# -------------------------
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found in .env")  # Comment tạm nếu chưa dùng

# -------------------------
# FastAPI router
# -------------------------
router = APIRouter()

# -------------------------
# Load embeddings & FAISS - SỬA LẠI
# -------------------------
try:
    df, faiss_handler = initialize_search_engine()
    print("✅ Search engine initialized successfully in API")
except Exception as e:
    print(f"❌ Error initializing search engine in API: {e}")
    df, faiss_handler = None, None

# -------------------------
# Input models
# -------------------------
class RecipeQuery(BaseModel):
    ingredients: List[str]

class SmoothQuery(BaseModel):
    top_dishes: List[dict]

# -------------------------
# API: search_recipes - SỬA LẠI
# -------------------------
@router.post("/search_recipes")
def search_recipes(query: RecipeQuery):
    """Search recipes by ingredients"""
    if not query.ingredients:
        return {"results": []}
    
    if df is None or faiss_handler is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        dishes = search_dishes(df, faiss_handler, query.ingredients, top_k=5)
        return {"results": dishes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# -------------------------
# API: smooth_recipes (Groq) - SỬA LẠI
# -------------------------
def smooth_recipes(query: SmoothQuery):
    """Generate description for top dishes"""
    if not query.top_dishes:
        return {"text": "Không có món ăn nào để hiển thị."}

    # Tạm thời trả về mô tả đơn giản nếu chưa dùng Groq
    dish_names = [dish["dish_name"] if isinstance(dish, dict) else dish for dish in query.top_dishes]
    
    if dish_names:
        text = f"Tìm thấy {len(dish_names)} món ăn phù hợp: {', '.join(dish_names)}. "
        text += "Bạn muốn thử món nào trong số này?"
    else:
        text = "Không tìm thấy món ăn phù hợp với nguyên liệu đã cung cấp."
    
    return {"text": text}

    # # Nếu muốn dùng Groq, bỏ comment phần dưới
    # try:
    #     groq_chat = ChatGroq(
    #         groq_api_key=GROQ_API_KEY,
    #         model_name="llama-3.1-8b-instant"
    #     )

    #     user_text = "\n".join([f"- {d}" for d in query.top_dishes])
        
    #     messages = [
    #         SystemMessage(
    #             content=(
    #                 "Bạn là chuyên gia ẩm thực Việt Nam. "
    #                 "Hãy trả lời trực tiếp, cuốn hút và thân thiện, dựa trên nguyên liệu mà người dùng cung cấp. "
    #                 "Không liệt kê công thức, không chào hỏi dài dòng. "
    #                 "Tập trung vào trải nghiệm khi ăn, mùi vị, màu sắc và cảm giác. "
    #                 "Bạn có thể gợi ý món ăn hoặc hỏi người dùng muốn thử món nào."
    #             )
    #         ),
    #         HumanMessage(
    #             content=(
    #                 f"Người dùng cung cấp nguyên liệu và tìm được các món ăn sau:\n{user_text}\n\n"
    #                 "Hãy viết đoạn chat như sau:\n"
    #                 "- Giới thiệu các món ăn phù hợp\n"
    #                 "- Mô tả cuốn hút, sinh động\n"
    #                 "- Có thể đưa ra gợi ý hoặc hỏi người dùng\n"
    #                 "Tối đa 3–5 câu, nhấn mạnh trải nghiệm, hương vị, màu sắc, cảm giác khi ăn."
    #             )
    #         )
    #     ]
        
    #     response = groq_chat.invoke(messages)  # Sửa từ generate thành invoke
    #     output_text = response.content
        
    #     return {"text": output_text}
        
    # except Exception as e:
    #     print(f"Groq error: {e}")
    #     return {"text": f"Tìm thấy các món ăn: {', '.join(query.top_dishes)}"}

# -------------------------
# API: smart_recipes (search + smooth) - SỬA LẠI
# -------------------------
@router.post("/smart_recipes")
def smart_recipes(query: RecipeQuery):
    """Smart search with AI description"""
    if not query.ingredients:
        return {"top_dishes": [], "description": "Vui lòng cung cấp nguyên liệu"}
    
    if df is None or faiss_handler is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    try:
        # 1️⃣ Lấy top dishes từ search
        top_dishes_results = search_dishes(df, faiss_handler, query.ingredients, top_k=5)

        # Debug: in ra terminal xem kết quả
        print("DEBUG - top_dishes_results:", [r["dish_name"] for r in top_dishes_results])

        # 2️⃣ Smooth bằng LLM/Groq
        smooth_result = smooth_recipes(SmoothQuery(top_dishes=top_dishes_results))

        return {
            "top_dishes": top_dishes_results,
            "description": smooth_result["text"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# -------------------------
# Health check endpoint
# -------------------------
@router.get("/health")
def health_check():
    """Health check for search API"""
    status = "healthy" if (df is not None and faiss_handler is not None) else "unhealthy"
    return {
        "status": status,
        "search_engine_initialized": df is not None and faiss_handler is not None
    }