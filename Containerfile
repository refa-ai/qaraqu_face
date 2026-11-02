# QARAQU FACE — Podman/Docker image (GPU host: --gpus all / nvidia CDI)
# Модели и gallery монтируются томом ./data
FROM docker.io/library/python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QARAQU_FACE_ENABLED=1 \
    QARAQU_FACE_HOST=0.0.0.0 \
    QARAQU_FACE_PORT=5020 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# На хосте с CUDA предпочтительно свой venv (facetrain cu128).
# В образе — torch cu121 wheel; для RTX 50xx лучше запускать нативно, не в CPU-slim.
RUN pip install --no-cache-dir -r /app/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu124

COPY config /app/config
COPY src /app/src
COPY web /app/web
COPY run.py /app/run.py
COPY PRINCIPLE.md /app/PRINCIPLE.md

RUN mkdir -p /app/data && \
    sed -i 's|root: /data/nimda/qaraqu_face|root: /app|' /app/config/default.yaml

EXPOSE 5020

CMD ["python", "/app/run.py"]
