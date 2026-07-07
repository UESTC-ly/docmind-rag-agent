# 用于容器化部署 API 和 Celery worker（共用同一镜像）
FROM python:3.12-slim

WORKDIR /app

# 系统依赖（psycopg2、pymupdf 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认起 API；worker 在 compose 里用 command 覆盖
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
