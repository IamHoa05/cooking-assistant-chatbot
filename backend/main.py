# import os
# from fastapi import FastAPI, APIRouter
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from app.api.recipes_api import router as api_router
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List

# app = FastAPI(title="Cooking Assistant API")

# # CORS nếu bạn dùng frontend riêng (chẳng hạn http://localhost:5173)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tạm thời cho localhost test
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount folder frontend/test1 để serve CSS/JS
# frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
# app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# # Include API routes
# app.include_router(api_router, prefix="/api")

# # Route trả index.html
# @app.get("/", response_class=HTMLResponse)
# def root():
#     html_path = os.path.join(frontend_path, "index.html")
#     with open(html_path, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content)

# # # Mount toàn bộ folder frontend để serve index.html + các file tĩnh nếu có
# # frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
# # app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# # # Include API routes
# # app.include_router(api_router, prefix="/api")

# # # Route trả index.html
# # @app.get("/", response_class=HTMLResponse)
# # def root():
# #     index_file = os.path.join(frontend_path, "index.html")
# #     with open(index_file, "r", encoding="utf-8") as f:
# #         html_content = f.read()
    
# #     # Thay thế đường dẫn file tĩnh nếu cần
# #     html_content = html_content.replace('src="', 'src="/static/')
# #     html_content = html_content.replace('href="', 'href="/static/')
    
# #     return HTMLResponse(html_content)








# app/main.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# import router từ api
from app.api import router as api_router

app = FastAPI(
    title="Vietnamese Recipe Search API",
    description="API tìm kiếm món ăn Việt Nam với nhiều tiêu chí: nguyên liệu, thể loại, độ khó, khẩu phần, thời gian nấu.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount router
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
def root():
    return {"message": "Vietnamese Recipe Search API is running."}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)



