# Sử dụng Python slim để nhẹ
FROM python:3.10-slim

# Set working directory trong container
WORKDIR /app

# Cài các dependencies hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer
COPY requirements.txt .

# Cài các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code backend
COPY . .

# Expose port FastAPI
EXPOSE 8000

# CMD dev: auto-reload khi sửa code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
