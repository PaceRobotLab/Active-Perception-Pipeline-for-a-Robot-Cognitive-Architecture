# ── Base: Stereolabs ZED SDK + CUDA + Python (Ubuntu 22.04) ───────────────────
# Includes: CUDA 12.1, ZED SDK 4.2, pyzed Python API pre-installed
FROM stereolabs/zed:4.2-py-devel-cuda12.1-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps for Open3D (OpenGL + X11) and OpenCV ─────────────────────────
# Note: libgl1-mesa-glx was renamed to libgl1 in Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
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

# SAM checkpoint mounted at runtime → /sam_vit_b_01ec64.pth
# Script resolves: os.path.dirname(__file__) = /app  → ../  = /
# So SAM_CKPT = /sam_vit_b_01ec64.pth  ✓

ENV PYTHONUNBUFFERED=1

CMD ["python", "vggt_guided_pipeline.py"]
