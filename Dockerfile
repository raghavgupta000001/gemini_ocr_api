# Use slim base
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable stdout/stderr line buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages (including tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtiff5-dev libjpeg62-turbo-dev zlib1g-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render provides $PORT env variable at runtime)
EXPOSE 8000

# Start uvicorn; allow PORT env variable provided by Render
CMD ["sh", "-lc", "uvicorn enhanced_ocr_with_grmini:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
