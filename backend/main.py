import os
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.api.recipes_api import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Cooking Assistant API")

# CORS nếu bạn dùng frontend riêng (chẳng hạn http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tạm thời cho localhost test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder frontend/test1 để serve CSS/JS
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Include API routes
app.include_router(api_router)

# Route trả index.html
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(frontend_path, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
