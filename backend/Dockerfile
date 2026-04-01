# backend/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for optimization solvers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpcre3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/saved_models

EXPOSE 8000

# Start the FastAPI Gateway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
