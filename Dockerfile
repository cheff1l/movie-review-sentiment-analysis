# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Set environment variables for clean container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies needed for compiling and running packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download required NLTK corpora during image build so the app runs offline
RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Copy application source code and models
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
COPY README.md .

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck to verify the web service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit service
ENTRYPOINT ["streamlit", "run", "app.py"]
