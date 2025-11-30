import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.recipes_api import router as api_router  # Import API router

from pydantic import BaseModel
from typing import List

# ============================================
# 1. Khởi tạo ứng dụng FastAPI
# ============================================
app = FastAPI(title="Cooking Assistant API")

# ============================================
# 2. Cấu hình CORS (Frontend gọi API)
# ============================================
# Trong giai đoạn phát triển: allow_origins = ["*"]
# Khi deploy thật: CHỈ cho phép domain cần thiết
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Cho phép frontend test localhost
    allow_credentials=True,
    allow_methods=["*"],              # Cho phép tất cả phương thức (GET/POST/PUT/DELETE)
    allow_headers=["*"],              # Cho phép mọi header
)

# ============================================
# 3. Mount thư mục frontend để serve static files
# ============================================
# Ví dụ: CSS, JS, images được đặt trong thư mục frontend/
# FastAPI sẽ truy cập bằng đường dẫn: /static/*
frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend")
)

app.mount(
    "/static",
    StaticFiles(directory=frontend_path),
    name="static"
)

# ============================================
# 4. Include API Routes (các endpoint chính)
# ============================================
app.include_router(api_router)

# ============================================
# 5. Serve file index.html tại route "/"
# ============================================
@app.get("/", response_class=HTMLResponse)
def root():
    """
    Trả về giao diện HTML chính của ứng dụng.
    Tự động load CSS/JS từ /static (đã mount ở trên).
    """
    html_path = os.path.join(frontend_path, "index.html")

    # Đảm bảo file tồn tại
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

    # Đọc nội dung file HTML và trả về
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)
