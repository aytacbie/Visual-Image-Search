# Container for small free hosts (Render free: 512 MB RAM). Runs as uid 1000 and serves on
# $PORT (Render sets it; defaults to 7860 for Hugging Face Spaces).
FROM python:3.12-slim

# int8 CLIP: ~200 MB RAM instead of ~500 MB, subcategory@1 81% instead of 84% on the catalogue.
ARG VS_MODEL_FILE=onnx/vision_model_quantized.onnx
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VS_MODEL_FILE=${VS_MODEL_FILE}

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

# Bake the model and the ONNX index into the image: startup is fast and needs no network.
# (An NV-CLIP index is built at first start instead, since the NVIDIA key is a runtime secret.)
RUN python scripts/ingest.py

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
