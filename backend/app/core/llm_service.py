# app/core/llm_service.py
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load env
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

def describe_dishes(top_dishes: List[str]) -> str:
    """
    Intent 'search': tạo đoạn mô tả hấp dẫn cho danh sách món ăn.
    """
    if not top_dishes:
        return "Không có món ăn nào để hiển thị."

    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia ẩm thực Việt Nam. "
            "Viết đoạn giới thiệu ngắn 3–5 câu mô tả sinh động, hấp dẫn về các món ăn dưới đây."
        )),
        HumanMessage(content=f"Các món ăn phù hợp:\n{dishes_text}")
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error (describe_dishes):", e)
        return "Không thể tạo mô tả món ăn bằng LLM."

def smooth_instructions(dish_name: str, ingredients: List[str], instructions: List[str]) -> str:
    """
    Intent 'detail': viết lại hướng dẫn nấu ăn ngắn gọn, dễ hiểu, hấp dẫn.
    """
    if not dish_name:
        return "Không có tên món ăn để tạo hướng dẫn."

    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

    ing_text = "\n".join([f"- {i}" for i in ingredients])
    step_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(instructions)])

    messages = [
        SystemMessage(content=(
            "Bạn là đầu bếp chuyên nghiệp. "
            "Viết lại hướng dẫn nấu ăn sao cho dễ hiểu, hấp dẫn, tối đa 3–5 câu."
        )),
        HumanMessage(content=(
            f"Món ăn: {dish_name}\n"
            f"Nguyên liệu:\n{ing_text}\n\n"
            f"Các bước nấu:\n{step_text}\n\n"
            "Hãy viết lại hướng dẫn mượt mà, tự nhiên, dễ làm theo."
        ))
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error (smooth_instructions):", e)
        return "Không thể tạo hướng dẫn mượt bằng LLM."
