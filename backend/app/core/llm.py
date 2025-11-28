# app/core/llm.py

from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load .env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -------------------------
# Hàm LLM Gen Z vui vẻ
# -------------------------
def create_chat():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )


def smooth_recipes(query):
    dishes = query.top_dishes or []
    if not dishes:
        return {"text": "Ối dồi ôi, chưa thấy món nào hợp luôn á. Thử thêm nguyên liệu khác nha!"}

    dishes_text = ", ".join(dishes)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là chuyên gia ẩm thực Việt Nam, giọng Gen Z, vui vẻ. "
                "Chỉ mô tả các món TRONG DANH SÁCH, không thêm món mới. Tối đa 50 chữ."
            ),
            HumanMessage(f"Danh sách món: {dishes_text}. Mô tả 1–2 món nổi bật nha.")
        ]
        resp = chat.invoke(messages)
        return {"text": resp.content}

    except:
        return {"text": f"Gợi ý mấy món này nè: {dishes_text}"}


def smooth_time_results(dishes: List[str]):
    if not dishes:
        return "Không thấy món nào hợp thời gian luôn á!"

    dishes_text = ", ".join(dishes)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là chuyên gia ẩm thực Gen Z. "
                "Mô tả vui vẻ, chọn 1–2 món trong danh sách. Tối đa 45 chữ."
            ),
            HumanMessage(f"Danh sách: {dishes_text}. Mô tả sinh động đi!")
        ]
        resp = chat.invoke(messages)
        return resp.content

    except:
        return f"Mấy món hợp thời gian: {dishes_text}"


def smooth_instructions(dish_name: str, instructions: List[str]):
    if not instructions:
        return "Không tìm thấy hướng dẫn nè!"

    steps = "\n".join(instructions)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là MasterChef Gen Z. Rewrite hướng dẫn nấu ăn trẻ trung, vui vẻ, "
                "nhưng vẫn rõ ràng. Tối đa 120 chữ."
            ),
            HumanMessage(f"Rewrite hướng dẫn món {dish_name}:\n{steps}")
        ]

        resp = chat.invoke(messages)
        return resp.content

    except:
        return steps


def smooth_category_results(category: str, dishes: List[str]):
    if not dishes:
        return f"Hông thấy món nào trong nhóm {category} luôn á!"

    dishes_text = ", ".join(dishes)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là reviewer ẩm thực Gen Z. "
                "Mô tả vui vẻ, chọn 1–2 món nổi bật. Tối đa 45 chữ."
            ),
            HumanMessage(f"Nhóm {category}. Danh sách: {dishes_text}. Mô tả món nổi bật nè.")
        ]
        resp = chat.invoke(messages)
        return resp.content

    except:
        return f"Mấy món trong nhóm {category}: {dishes_text}"


def smooth_difficulty_results(difficulty: str, dishes: List[str]):
    if not dishes:
        return f"Hông thấy món level {difficulty} luôn á!"

    dishes_text = ", ".join(dishes)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là MasterChef Gen Z. "
                "Mô tả 1–2 món nổi bật, vui vẻ, trẻ trung. Tối đa 45 chữ."
            ),
            HumanMessage(f"Độ khó {difficulty}. Danh sách: {dishes_text}. Mô tả món nổi bật nha.")
        ]
        resp = chat.invoke(messages)
        return resp.content

    except:
        return f"Mấy món độ khó {difficulty}: {dishes_text}"


def smooth_servings_results(servings: int, dishes: List[str]):
    if not dishes:
        return f"Không thấy món nào hợp khẩu phần {servings} người luôn á!"

    dishes_text = ", ".join(dishes)

    try:
        chat = create_chat()
        messages = [
            SystemMessage(
                "Bạn là food reviewer Gen Z. "
                "Mô tả 1–2 món sinh động, vui vẻ. Tối đa 45 chữ."
            ),
            HumanMessage(
                f"Khẩu phần {servings} người. Danh sách: {dishes_text}. Mô tả món nổi bật giùm."
            )
        ]
        resp = chat.invoke(messages)
        return resp.content

    except:
        return f"Mấy món hợp {servings} người: {dishes_text}"
