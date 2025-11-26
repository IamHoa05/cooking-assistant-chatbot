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

def smooth_dishes(top_dishes: List[str]) -> str:
    """Tạo đoạn giới thiệu hấp dẫn cho danh sách món ăn"""
    if not top_dishes:
        return "Không có món ăn nào để hiển thị."

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia ẩm thực Việt Nam. "
            "Viết đoạn giới thiệu ngắn (3–5 câu), mô tả sinh động về các món phù hợp."
        )),
        HumanMessage(content=(
            f"Các món phù hợp với nguyên liệu bạn cung cấp:\n{dishes_text}\n"
            "Hãy mô tả hấp dẫn, sinh động và tự nhiên."
        ))
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error:", e)
        return "Không thể tạo mô tả món ăn bằng LLM."


def smooth_instructions(dish_name: str, ingredients: list, instructions: list) -> str:
    """Viết lại hướng dẫn nấu ăn ngắn gọn, dễ hiểu, hấp dẫn"""
    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

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
            "Hãy viết lại hướng dẫn mượt và tự nhiên."
        ))
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error:", e)
        return "Không thể tạo hướng dẫn mượt bằng LLM."

def smooth_dishes_by_difficulty(top_dishes: List[str], difficulty: str) -> str:
    """LLM giới thiệu món ăn theo độ khó (dễ / trung bình)"""
    if not top_dishes:
        return "Không có món ăn nào phù hợp."

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia ẩm thực Việt Nam. "
            f"Viết đoạn giới thiệu ngắn (3–5 câu) về các món ăn '{difficulty}' "
            "với mô tả sinh động, hấp dẫn và thân thiện."
        )),
        HumanMessage(content=(
            f"Các món ăn thuộc độ khó '{difficulty}':\n{dishes_text}\n"
            "Hãy mô tả cảm giác khi ăn, màu sắc, hương vị và trải nghiệm của món ăn."
        ))
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error:", e)
        return f"Không thể tạo mô tả món ăn '{difficulty}' bằng LLM."
    
def smooth_by_cooking_time(top_dishes, time_range):
    """
    Mô tả món ăn dựa trên thời gian nấu.
    """
    if not top_dishes:
        return f"Không có món ăn trong khoảng {time_range} phút."
    groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    dishes_text = "\n".join([f"- {d}" for d in top_dishes])
    messages = [
        SystemMessage(content=f"Bạn là chuyên gia ẩm thực. Mô tả món ăn thời gian nấu {time_range} phút, 3-5 câu."),
        HumanMessage(content=f"Các món nấu trong khoảng {time_range} phút:\n{dishes_text}\nHãy mô tả sinh động, hấp dẫn.")
    ]
    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ LLM error (cooking time):", e)
        return f"Không thể tạo mô tả món ăn trong khoảng {time_range} phút."
    
def smooth_by_servings(top_dishes: List[str], context: str = "khẩu phần") -> str:
    """
    Smooth mô tả món ăn bằng LLM/Groq.
    context: có thể là 'khẩu phần', 'thời gian nấu', 'độ khó', ...
    """
    if not top_dishes:
        return f"Không có món ăn phù hợp với {context} yêu cầu."

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia ẩm thực Việt Nam. "
                f"Viết đoạn giới thiệu ngắn (3–5 câu) mô tả sinh động về các món ăn phù hợp với {context}."
            )
        ),
        HumanMessage(
            content=(
                f"Các món phù hợp:\n{dishes_text}\n"
                "Hãy mô tả hấp dẫn, sinh động và tự nhiên để gợi cảm giác ngon miệng."
            )
        )
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ Groq LLM error:", e)
        return "Không thể tạo mô tả món ăn bằng LLM."
    
def smooth_by_category(top_dishes: list, category: str):
    """
    Dùng LLM/Groq để tạo mô tả hấp dẫn cho danh sách món theo thể loại.
    """
    if not top_dishes:
        return "Không có món ăn nào để hiển thị."

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(content=(
            "Bạn là chuyên gia ẩm thực Việt Nam. "
            "Viết đoạn giới thiệu 3–5 câu sinh động, hấp dẫn về các món ăn phù hợp."
        )),
        HumanMessage(content=(
            f"Người dùng muốn xem các món thể loại: {category}\n"
            f"Danh sách các món ăn:\n{dishes_text}\n"
            "Hãy mô tả sinh động, cuốn hút và tự nhiên."
        ))
    ]

    try:
        response = groq_chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print("⚠️ Groq LLM error (category):", e)
        return "Không thể tạo mô tả món ăn bằng LLM."