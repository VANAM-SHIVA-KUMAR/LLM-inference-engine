# -----------------------------------------------------------------------
# GPU-Accelerated LLM Inference Engine
# Base: CUDA 12.1 + Python 3.11
# Optimizations: INT8 quantization (bitsandbytes) + Flash Attention 2
# -----------------------------------------------------------------------

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install Flash Attention 2 (needs CUDA headers)
RUN pip install flash-attn --no-build-isolation

# Copy source
COPY . .

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start FastAPI server
CMD ["python", "server.py"]
