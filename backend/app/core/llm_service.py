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

def describe_dishes(top_dishes: List[str], user_input: str) -> str:
    """
    Tạo mô tả món ăn dựa trên danh sách món đã được lọc.
    Kết quả: luôn trả về 1 string.
    """
    if not top_dishes:
        return "Hông thấy món nào hợp á, thử thêm nguyên liệu khác xem!"

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    # Format danh sách món
    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia ẩm thực Việt Nam, giọng Gen Z.\n"
                "LUẬT BẮT BUỘC – LLM PHẢI TUÂN THỦ:\n"
                "1) Chỉ được mô tả các món trong DANH SÁCH MÓN.\n"
                "2) Không được tự nghĩ thêm món mới hoặc nhắc món ngoài danh sách.\n"
                "3) Có thể liên hệ nhẹ đến yêu cầu của người dùng, nhưng KHÔNG được tạo món mới từ input.\n"
                "4) Không được nói chung chung kiểu 'món Việt Nam' – phải nhắc tên món trong danh sách.\n"
                "5) Chọn 1–2 món trong danh sách để mô tả.\n"
                "6) Mô tả tự nhiên, vui vẻ, tối đa 40 chữ.\n"
                "7) Không emoji. Không hỏi người dùng."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu ban đầu của người dùng: {user_input}\n"
                f"DANH SÁCH MÓN: \n{dishes_text}\n"
                "Hãy mô tả 1–2 món trong danh sách sao cho hợp ngữ cảnh yêu cầu trên."
            )
        )
    ]

    try:
        response = groq_chat.generate([messages])
        text = response.generations[0][0].text.strip()
        return text
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
