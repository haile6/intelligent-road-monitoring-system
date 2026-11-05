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

# Copy dependency manifest and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy complete project codebase
COPY . .

# Ensure storage directories for MLOps registry, runs, and outputs exist
RUN mkdir -p data/mlops/registry data/mlops/runs outputs/plate_crops outputs/reports

# Expose Web Interface & MLOps Serving API ports
EXPOSE 5000 8080

# Healthcheck probe against MLOps serving API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Launch RoadEye Production Server
CMD ["python", "web/app.py"]
