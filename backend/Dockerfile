FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY alembic.ini .
COPY alembic ./alembic
COPY ake_backend ./ake_backend
CMD ["uvicorn", "ake_backend.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
