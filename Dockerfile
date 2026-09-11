FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip setuptools wheel

# Install dependencies in modular layers with retries to prevent connection drops & enable layer caching
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 10 fastapi uvicorn pydantic requests joblib plotly
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 10 numpy pandas
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 10 scikit-learn
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 10 xgboost
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 10 streamlit

# Copy requirements.txt for reference
COPY requirements.txt .

# Copy backend and frontend application source code and models
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501
