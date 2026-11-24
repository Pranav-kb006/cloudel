# Serverless LLM Progressive Layer-Streaming Worker Container
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STORAGE_BACKEND=rangeserver \
    BANDWIDTH_MBPS=250

WORKDIR /app

# Install CPU PyTorch + streaming engine dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    aiohttp>=3.9 \
    psutil>=5.9 \
    requests>=2.31 \
    numpy \
    matplotlib

COPY . /app

# Generate the default 12-layer Llama BF16 safetensors checkpoint if absent
RUN python3 src/bench/make_transformer_model.py \
    --out /app/models/llama_12layer_140m.safetensors

EXPOSE 8000 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status')" || exit 1

CMD ["python3", "web/demo_server.py"]
