


# app/api.py
from xml.sax import handler
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
import numpy as np
import requests

# utils
from app.utils.faiss_handler import FAISSHandler
from app.utils.embedder import load_vietnamese_embedding_model, embed_texts
from langchain_groq import ChatGroq  # hoặc gemini API client
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.search_engine import search_dishes

# -------------------------
# Load .env
# -------------------------
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# -------------------------
# FastAPI router
# -------------------------
router = APIRouter()

# -------------------------
# Load embeddings & FAISS
# -------------------------
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

# -------------------------
# Input models
# -------------------------
class RecipeQuery(BaseModel):
    ingredients: List[str]

class SmoothQuery(BaseModel):
    top_dishes: List[str]

# -------------------------
# API: search_recipes
# -------------------------
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

#     output = [{"dish_name": r.get("dish_name")} for r in results]
#     return {"results": output}

def search_recipes(query: RecipeQuery):
    dishes = search_dishes(df, faiss_handler, query.ingredients)
    return {"results": dishes}

# -------------------------
# API: smooth_recipes (Groq)
# -------------------------

def smooth_recipes(query: SmoothQuery):
    if not query.top_dishes:
        return {"text": "Không có món ăn nào để hiển thị."}

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"  # hoặc model Groq/Gemini bạn muốn
    )

    user_text = "\n".join([f"- {d}" for d in query.top_dishes])
    

    messages = [
    SystemMessage(
        content=(
            "Bạn là chuyên gia ẩm thực Việt Nam. "
            "Hãy trả lời trực tiếp, cuốn hút và thân thiện, dựa trên nguyên liệu mà người dùng cung cấp. "
            "Không liệt kê công thức, không chào hỏi dài dòng. "
            "Tập trung vào trải nghiệm khi ăn, mùi vị, màu sắc và cảm giác. "
            "Bạn có thể gợi ý món ăn hoặc hỏi người dùng muốn thử món nào."
        )
    ),
    HumanMessage(
        content=(
            "Người dùng cung cấp nguyên liệu sau: {user_text}\n\n"
            "Hãy viết đoạn chat như sau:\n"
            "- Giới thiệu các món ăn phù hợp\n"
            "- Mô tả cuốn hút, sinh động\n"
            "- Có thể đưa ra gợi ý hoặc hỏi người dùng\n"
            "Tối đa 3–5 câu, nhấn mạnh trải nghiệm, hương vị, màu sắc, cảm giác khi ăn."
        )
    )
]
    response = groq_chat.generate([messages])  # lưu ý: phải bọc trong list!
    output_text = response.generations[0][0].text

    return {"text": output_text}

# -------------------------
# API: smart_recipes (search + smooth)
# -------------------------
@router.post("/smart_recipes")
def smart_recipes(query: RecipeQuery):
    # 1️⃣ Lấy top dishes từ hàm logic search_dishes
    top_dishes = search_dishes(df, faiss_handler, query.ingredients)

    # Debug: in ra terminal xem kết quả
    print("DEBUG - top_dishes:", top_dishes)

    # 2️⃣ Smooth bằng LLM/Groq
    smooth_result = smooth_recipes(SmoothQuery(top_dishes=top_dishes))

    return {
        "top_dishes": top_dishes,           # <-- optional trả về frontend luôn
        "description": smooth_result["text"]
    }