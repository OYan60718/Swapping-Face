FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    git ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libegl1 \
    libgles2-mesa \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

WORKDIR /workspace
