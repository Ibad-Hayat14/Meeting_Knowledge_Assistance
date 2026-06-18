# Dockerfile for FastAPI Backend
FROM python:3.11-slim

# Install system dependencies (FFmpeg is required for audio extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src/ /app/src/

# Expose the API port
EXPOSE 8000

# Define runtime environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Start Uvicorn bound to host and Render dynamic port
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
