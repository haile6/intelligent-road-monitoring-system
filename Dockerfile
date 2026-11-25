# ============================================================
# RoadEye Production MLOps Container
# Multi-Stage, Production-Grade Container for AI & Video Telemetry
# ============================================================

FROM python:3.10-slim

# Prevent interactive prompts & tune Python execution
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1

# Install system dependencies required for OpenCV, FFmpeg, and audio synthesis
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest and install CPU-only PyTorch first to keep image lightweight for Railway
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy complete project codebase
COPY . .

# Ensure storage directories for MLOps registry, runs, database, and outputs exist
RUN mkdir -p data/mlops/registry data/mlops/runs outputs/plate_crops outputs/reports database

# Default port (Railway overrides PORT dynamically at runtime)
ENV PORT=8080
EXPOSE 8080

# Healthcheck probe against MLOps serving API
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/v1/health || exit 1

# Launch RoadEye Production Gunicorn Server bound to Railway's $PORT
CMD ["sh", "-c", "gunicorn web.app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 300"]
