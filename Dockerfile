FROM python:3.12.13-slim

# OCR, audio decoding, and shared libraries needed by Pillow/OpenCV-backed tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Decision 11 pins every Python dependency used by the deployment image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Avoid a model download during a Render cold start or the first audio request.
ENV HF_HOME=/opt/huggingface
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# This copies app/static/ with the API, so Render serves one frontend/API image.
COPY app ./app
COPY prompts ./prompts

EXPOSE 8000

# Render supplies PORT; the fallback keeps docker run -p 8000:8000 convenient.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
