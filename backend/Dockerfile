# backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/saved_models

EXPOSE 8000

# Apply migrations, then start the FastAPI gateway.
CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]
