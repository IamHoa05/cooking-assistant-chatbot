# app/core/llm_service.py
"""
LLM Service
-----------
Module sử dụng LangChain Groq (ChatGroq) để:
1. Mô tả món ăn dựa trên danh sách món được lọc
2. Rewrite hướng dẫn nấu ăn ngắn gọn, trẻ trung, dễ hiểu
"""

from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# =====================================
# 1. Load environment variables
# =====================================
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# =====================================
# 2. Describe dishes
# =====================================
def describe_dishes(top_dishes: List[str], user_input: str) -> str:
    """
    Tạo mô tả món ăn dựa trên danh sách món đã được lọc.
    Luôn trả về 1 string.
    
    Args:
        top_dishes: danh sách tên món được lọc
        user_input: yêu cầu gốc của người dùng

    Returns:
        str: mô tả 1–2 món, hoặc hỏi lại user nếu cần
    """
    if not top_dishes:
        return "Hông thấy món nào hợp á, thử thêm nguyên liệu khác xem!"

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    # Format danh sách món
    dishes_text = "\n".join([f"- {d}" for d in top_dishes])

    # Messages gửi cho LLM
    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia ẩm thực Việt Nam, giọng Gen Z.\n"
                "LUẬT BẮT BUỘC – LLM PHẢI TUÂN THỦ:\n"
                "1) Chỉ được mô tả hoặc hỏi về các món trong DANH SÁCH MÓN.\n"
                "2) Không được tự nghĩ thêm món mới hoặc nhắc món ngoài danh sách.\n"
                "3) Có thể liên hệ nhẹ đến yêu cầu của người dùng, nhưng KHÔNG được tạo món mới từ input.\n"
                "4) Không được nói chung chung kiểu 'món Việt Nam' – phải nhắc tên món trong danh sách.\n"
                "5) Khi mô tả, chỉ được chọn 1–2 món trong danh sách.\n"
                "6) Mô tả tự nhiên, vui vẻ, tối đa 40 chữ.\n"
                "7) Chỉ được hỏi lại người dùng 1 câu DUY NHẤT khi cần họ chọn món nào trong DANH SÁCH MÓN.\n"
                "8) KHÔNG được hỏi bất kỳ thông tin nào khác ngoài việc chọn 1 món trong danh sách.\n"
                "9) Không emoji."
            )
        ),
        HumanMessage(
            content=(
                f"Yêu cầu ban đầu của người dùng: {user_input}\n"
                f"DANH SÁCH MÓN:\n{dishes_text}\n"
                "Hãy mô tả 1–2 món, hoặc hỏi lại người dùng muốn hướng dẫn món nào."
            )
        )
    ]

    # Gọi LLM
    try:
        response = groq_chat.generate([messages])
        text = response.generations[0][0].text.strip()
        return text
    except Exception as e:
        print("⚠️ LLM error (describe_dishes):", e)
        return "Không thể tạo mô tả món ăn bằng LLM."


# =====================================
# 3. Smooth cooking instructions
# =====================================
def smooth_instructions(dish_name: str, ingredients: List[str], instructions: List[str]) -> str:
    """
    Rewrite hướng dẫn nấu ăn ngắn gọn, dễ hiểu, trẻ trung.
    Không đánh số thứ tự, không bôi đen, format sạch đẹp. Tối đa 120 chữ.

    Args:
        dish_name: tên món
        ingredients: danh sách nguyên liệu
        instructions: hướng dẫn gốc (list các bước)

    Returns:
        str: hướng dẫn được rewrite
    """
    if not dish_name:
        return "Không có tên món ăn để tạo hướng dẫn."

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    # Chuẩn bị text cho LLM
    ing_text = "\n".join([f"- {i}" for i in ingredients])
    inst_text = "\n".join(instructions)

    messages = [
        SystemMessage(
            content=(
                "Bạn là MasterChef Gen Z. Viết lại hướng dẫn nấu ăn ngắn gọn, trẻ trung, vui vẻ "
                "nhưng rõ ràng. KHÔNG ĐƯỢC ĐÁNH SỐ BẤT KỲ BƯỚC NÀO, không bôi đen, không thêm ký tự thừa. "
                "Chỉ tạo phần hướng dẫn, không liệt kê nguyên liệu. Format sạch đẹp, tối đa 120 chữ."
            )
        ),
        HumanMessage(
            content=(
                f"Món: {dish_name}\n"
                f"Nguyên liệu:\n{ing_text}\n\n"
                f"Instructions gốc:\n{inst_text}\n\n"
                "Hãy rewrite lại hướng dẫn nấu ăn theo yêu cầu trên, KHÔNG đánh số thứ tự."
            )
        )
    ]

    # Gọi LLM
    try:
        resp = groq_chat.invoke(messages)
        text = resp.content.strip()

        # Tách thành list, loại bỏ các ký tự đánh số còn sót
        steps = [
            line.strip().lstrip("• ").lstrip("- ").replace("– ", "").strip()
            for line in text.split("\n")
            if line.strip()
        ]

        return steps
    except Exception as e:
        print("⚠️ LLM error (smooth_instructions):", e)
        return "Không thể tạo hướng dẫn mượt bằng LLM."
