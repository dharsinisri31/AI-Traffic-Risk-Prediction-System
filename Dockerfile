# Use official Python 3.11 slim image for a lightweight, secure container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set container working directory
WORKDIR /app

# Install system dependencies if required for compiling C/C++ packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy backend application, trained pickle models, and frontend code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (will be overridden by docker-compose for each specific service)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
