# Active Perception Pipeline for a Robot Cognitive Architecture

Real-time 3D scene reconstruction and object detection pipeline integrating ZED stereo vision, VGGT multiview 3D inference, Grounding DINO, and SAM — governed by an ADAPT cognitive architecture that drives active camera sweeps, evaluates detections via confidence-gated port states, and builds labeled point cloud representations of the environment.

---

## Overview

The pipeline takes live stereo frames from a ZED camera, reconstructs the full 3D scene using Facebook's VGGT-1B model, detects user-specified objects with Grounding DINO, segments them precisely with SAM, and visualizes the labeled point cloud in Open3D — all guided by an ADAPT goal stack that classifies each detection as `verified`, `tentative`, or `rejected`.

---

## Pipeline Flow

```
ZED Left Lens (BGR 1280×720)
  └─► BGR→RGB + center-crop 720×720 → save PNG    [60 frames @ 0.4s intervals]
        └─► subsample to 9 evenly-spaced frames
              └─► VGGT-1B inference
                    ├─ world_points  (9, H, W, 3)   — absolute XYZ per pixel (metres)
                    └─ world_conf    (9, H, W)       — per-pixel confidence
                          └─► Grounding DINO  (5 frames, text prompt)
                                └─► SAM segmentation  (anchor frame + DINO bbox)
                                      └─► ADAPT port_state  (verified / tentative / rejected)
                                            └─► build_scene  (filter + color + voxel + outlier removal)
                                                  └─► Open3D 3D Visualizer
```

---

## ADAPT Cognitive Architecture

| Component | Value |
|---|---|
| Goal type | `find-object` |
| Operator | `investigate-discrepancy` |
| TAU_HIGH | `0.40` → `verified` |
| TAU_LOW | `0.15` → `tentative` |
| Below TAU_LOW | `rejected` |

Each detected object is stored as an **RS schema** (Representational Schema) with:
- 3D pose (XYZ centroid from VGGT world points)
- Confidence score
- Port state (`verified` / `tentative` / `rejected`)
- Next cognitive action (`accept_and_insert` / `insert_tentative_schedule_resaccade` / `reject`)

Results are saved as JSON to the `results/` folder after each run.

---

## Hardware Requirements

- **ZED Stereo Camera** (any model supporting HD720)
- **NVIDIA GPU** with CUDA (VGGT-1B requires ~6GB VRAM minimum; 8GB+ recommended)
- Windows or Linux

---

## Models Required

The pipeline uses three external models. None of them are included in this repo — you must download/install them before running.

| Model | Source | Size | How it loads |
|---|---|---|---|
| **VGGT-1B** | HuggingFace `facebook/VGGT-1B` | ~4 GB | Auto-downloaded on first run via `VGGT.from_pretrained()` |
| **Grounding DINO** | HuggingFace `IDEA-Research/grounding-dino-tiny` | ~340 MB | Auto-downloaded on first run via `AutoModel.from_pretrained()` |
| **SAM ViT-B** | Meta (manual download) | ~358 MB | Must be placed manually — see Step 3 below |

> VGGT and DINO are pulled from HuggingFace automatically the first time you run the script (requires internet). SAM must be downloaded manually once and placed at the correct path.

---

## Installation

### 1. ZED SDK

Download and install the ZED SDK from [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release), then register the Python API:

```bash
python "C:/Program Files (x86)/ZED SDK/get_python_api.py"
```

### 2. Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python Pillow open3d transformers accelerate

# VGGT (Facebook Research)
pip install git+https://github.com/facebookresearch/vggt.git

# Segment Anything (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> See `requirements.txt` for the full pinned list.

### 3. SAM checkpoint (manual download required)

SAM weights are not on HuggingFace — download the ViT-B checkpoint directly from Meta and place it **one level above** this repo folder:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile ..\sam_vit_b_01ec64.pth

# Linux / Mac
wget -O ../sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Expected path: `../sam_vit_b_01ec64.pth` relative to `vggt_guided_pipeline.py`.

If you prefer a different location, update this line in the script:
```python
SAM_CKPT = os.path.join(os.path.dirname(__file__), "..", "sam_vit_b_01ec64.pth")
```

### 4. First-run model downloads

On the very first run, VGGT (~4 GB) and Grounding DINO (~340 MB) will be downloaded automatically from HuggingFace and cached locally. This only happens once. Make sure you have a stable internet connection and enough disk space (~5 GB free).

---

## Usage

```bash
python vggt_guided_pipeline.py
```

**On launch:**
1. Enter the objects you want to find (comma-separated), e.g.: `chairs, screens, bag`
2. Sweep the ZED camera **left to right** slowly — 60 frames are auto-captured over ~24 seconds
3. VGGT, DINO, and SAM run automatically
4. Open3D window opens with the labeled 3D reconstruction

**Controls during capture:**
- `Q` — quit at any time

---

## Configuration

Key parameters at the top of `vggt_guided_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `CAPTURE_INTERVAL` | `0.4s` | Time between auto-captured frames |
| `MAX_FRAMES` | `60` | Total ZED frames to capture |
| `VGGT_MAX_FRAMES` | `9` | Frames subsampled for VGGT |
| `DINO_BOX_THR` | `0.25` | Grounding DINO box confidence threshold |
| `TAU_HIGH` | `0.40` | ADAPT verified threshold |
| `TAU_LOW` | `0.15` | ADAPT tentative threshold |
| `VOXEL_SIZE` | `0.004` | Point cloud voxel size (4mm) |
| `MAX_SCENE_PTS` | `1,500,000` | Max points in scene before subsampling |

---

## Output

After each run, results are saved to `results/detection_YYYYMMDD_HHMMSS.json`:

```json
{
  "goal": ["chairs", "screens"],
  "detections": [
    {
      "schema_type": "RS",
      "class": "chairs",
      "port_state": "verified",
      "confidence": 0.642,
      "pose": {"x": 0.12, "y": -0.45, "z": 1.83},
      "next_action": "accept_and_insert",
      "image_region": {"x1": 120.0, "y1": 80.0, "x2": 480.0, "y2": 560.0, "frame": 5}
    }
  ],
  "timing_s": {
    "vggt_infer_s": 94.2,
    "dino_infer_s": 3.1,
    "sam_infer_s": 1.8,
    "total_s": 142.5
  }
}
```

---

## Folder Structure

```
.
├── vggt_guided_pipeline.py   # Main pipeline
├── requirements.txt
├── recordings/               # Place .avi files here for batch inference
└── results/                  # JSON detection outputs saved here automatically
```

---

## Citation

If you use this pipeline in your research, please cite:

- **VGGT**: Wang et al., *VGGT: Visual Geometry Grounded Transformer*, Facebook Research, 2024
- **Grounding DINO**: Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training*, IDEA Research, 2023
- **SAM**: Kirillov et al., *Segment Anything*, Meta AI, 2023
- **ZED SDK**: Stereolabs

---

## License

This project is for research purposes. VGGT, SAM, and Grounding DINO are subject to their respective licenses (see their repositories).
