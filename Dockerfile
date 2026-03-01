FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt update && apt install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
