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


# Load .env
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# FastAPI router
router = APIRouter()



# Load embeddings & FAISS
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



# Input models
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

def smooth_recipes(query: SmoothQuery):
    dishes = query.top_dishes or []
    if not dishes:
        return {"text": "Hông thấy món nào hợp á, thử thêm nguyên liệu khác xem!"}

    dishes_text = ", ".join(dishes)

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    messages = [
    SystemMessage(
        content=(
            "Bạn là chuyên gia ẩm thực Việt Nam, giọng Gen Z, vui vẻ.\n"
            "LUẬT BẮT BUỘC:\n"
            "1) Chỉ nói về các món trong DANH SÁCH HỆ THỐNG cung cấp.\n"
            "2) Không bịa món hay thêm món ngoài danh sách.\n"
            "3) Trả lời ≤ 50 chữ.\n"
            "4) Bạn được phép gợi ý 1-2 món nổi bật nên thử trong danh sách, bằng cách mô tả thật tự nhiên và sinh động cảm giác khi ăn.\n"
            "5) Không hỏi người dùng, không gợi ý nguyên liệu ngoài danh sách."
        )
    ),
    SystemMessage(
        content=(
            "Ví dụ:\n"
            "Danh sách món: Phở bò, Bò xào sả ớt, Gỏi cuốn\n"
            "Đúng: 'Phở bò thơm nức mùi quế hồi, ăn là mê luôn. Bò xào sả ớt cay nhẹ, siêu đã!'"
            "❌ Sai: 'Món ăn thơm ngon, cuốn hút lắm.' (không nhắc tên món)"
        )
    ),
    HumanMessage(
        content=(
            f"DANH SÁCH MÓN HIỆN TẠI: {dishes_text}\n"
            "Hãy mô tả thật tự nhiên và gợi ý 1-2 món nổi bật nên thử. "
            "Không thêm món mới, không đặt câu hỏi, tối đa 40 chữ."
        )
    )
]



    response = groq_chat.invoke(messages)
    text = response.content.replace("?", "")  # loại bỏ câu hỏi nếu có
    return {"text": text}




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