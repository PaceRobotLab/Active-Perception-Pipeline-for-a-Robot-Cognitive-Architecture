# ── Base: Stereolabs ZED SDK + CUDA + Python (Ubuntu 22.04) ───────────────────
# Includes: CUDA 12.1, ZED SDK 4.2, pyzed Python API pre-installed
FROM stereolabs/zed:4.2-py-devel-cuda12.1-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps for Open3D display and OpenCV ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (CUDA 12.1) ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── Core Python dependencies ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    Pillow \
    open3d \
    transformers \
    accelerate

# ── VGGT (Facebook Research) ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/vggt.git

# ── Segment Anything (SAM) ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/segment-anything.git

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app

COPY vggt_guided_pipeline.py .
COPY results/ results/

# SAM checkpoint is mounted at runtime — expected at /sam_vit_b_01ec64.pth
# (one level above /app, matching the path in the script)
# HuggingFace model cache is also mounted to avoid re-downloading each run

ENV PYTHONUNBUFFERED=1

CMD ["python", "vggt_guided_pipeline.py"]
