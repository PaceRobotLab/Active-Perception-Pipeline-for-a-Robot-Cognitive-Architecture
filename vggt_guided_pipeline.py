"""
ZED capture -> VGGT 3D reconstruction -> Grounding DINO + SAM -> Open3D.

Sweep camera LEFT to RIGHT, press SPACE at each position to capture a frame.
After N frames, VGGT reconstructs the scene and DINO+SAM highlights your objects.

Controls (during capture):
  SPACE = capture current frame
  Q     = quit
"""

import os, tempfile, shutil, time, json, datetime, uuid
import cv2
import numpy as np
import torch
import open3d as o3d
import pyzed.sl as sl
from PIL import Image as PILImage

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Startup ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VGGT ACTIVE PERCEPTION  [basetest3]")
print("=" * 60)

raw_goal = input("  What objects should I find? (comma-separated)\n  > ").strip()
QUERY_CLASSES = [c.strip() for c in raw_goal.split(",") if c.strip()] or ["object"]
T_GOAL = time.time()

# ── ADAPT goal stack ──────────────────────────────────────────
ADAPT_GOAL = {
    "goal_type":      "find-object",
    "target_classes": QUERY_CLASSES,
    "operator":       "investigate-discrepancy",
    "preconditions":  {
        "sweep_complete":  False,   # set True after 60 frames
        "focal_roi_set":   False,   # set True when DINO finds a box
        "reconstruction_ready": False,
    },
}
print(f"\n[ADAPT] Goal stack : {{'goal_type': 'find-object', 'targets': {QUERY_CLASSES}}}")
print(f"[ADAPT] Operator   : investigate-discrepancy")
print(f"[ADAPT] Precond    : sweep_complete=False  focal_roi_set=False")
print(f"\n  Keep sweeping LEFT ↔ RIGHT — 60 frames auto-captured, then VGGT runs.")
print(f"  Q = quit.")
print("=" * 60 + "\n")

# ─── Config ───────────────────────────────────────────────────────────────────
CAPTURE_INTERVAL  = 0.4    # seconds between frames (~2.5 fps → 60 frames in ~24 s)
MAX_FRAMES        = 60     # capture exactly this many, then auto-infer
VGGT_MAX_FRAMES   = 9      # subsample to this before VGGT (keeps inference <3 min)
MAX_SCENE_PTS     = 1_500_000
VOXEL_SIZE        = 0.004   # 4mm voxels — denser surface
DINO_REPO         = "IDEA-Research/grounding-dino-tiny"
DINO_BOX_THR      = 0.25
DINO_TEXT_THR     = 0.20
MAX_SAM_PER_CLASS = 12
TAU_HIGH          = 0.40   # port_state: verified
TAU_LOW           = 0.15   # port_state: tentative (below = rejected)
SAM_CKPT          = os.path.join(os.path.dirname(__file__), "..", "sam_vit_b_01ec64.pth")
SAM_TYPE          = "vit_b"

_PALETTE = [
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 0.3, 0.0]),
    np.array([0.0, 0.5, 1.0]),
    np.array([1.0, 0.0, 0.8]),
    np.array([1.0, 1.0, 0.0]),
]
CLASS_COLORS  = {cls.lower(): _PALETTE[i % len(_PALETTE)] for i, cls in enumerate(QUERY_CLASSES)}
DEFAULT_COLOR = np.array([0.0, 1.0, 1.0])

_COLOR_NAMES = ["green", "orange", "blue", "pink", "yellow"]

def colour_name(cls):
    idx = QUERY_CLASSES.index(cls) if cls in QUERY_CLASSES else -1
    return _COLOR_NAMES[idx % len(_COLOR_NAMES)] if idx >= 0 else "cyan"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_port_state(confidence):
    if confidence >= TAU_HIGH:
        return "verified"
    elif confidence >= TAU_LOW:
        return "tentative"
    return "rejected"

def extract_pose_xyz(world_points, mask_wp, frame_idx):
    """Return (x, y, z) centroid of world_points at SAM-mask pixels."""
    pts_frame = world_points[frame_idx]          # (H, W, 3)
    mask_bool = mask_wp > 0.5
    if mask_bool.sum() == 0:
        return None
    pts_masked = pts_frame[mask_bool]            # (N, 3)
    valid = np.isfinite(pts_masked).all(axis=1)
    if valid.sum() == 0:
        return None
    c = pts_masked[valid].mean(axis=0)
    return {"x": round(float(c[0]), 4), "y": round(float(c[1]), 4), "z": round(float(c[2]), 4)}

def cognitive_feedback(port_state, cls):
    if port_state == "verified":
        action = "accept_and_insert"
        msg    = f"  → VERIFIED   : accept '{cls}' — inserting into working memory"
    elif port_state == "tentative":
        action = "insert_tentative_schedule_resaccade"
        msg    = f"  → TENTATIVE  : '{cls}' uncertain — scheduling re-saccade for confirmation"
    else:
        action = "reject"
        msg    = f"  → REJECTED   : '{cls}' confidence too low — discarding"
    return action, msg

def save_results_json(detection_records, not_found, timing):
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp":   ts,
        "adapt_goal":  ADAPT_GOAL,
        "goal":        QUERY_CLASSES,
        "detections":  detection_records,
        "not_found":   not_found,
        "timing_s":    timing,
    }
    path = os.path.join(RESULTS_DIR, f"detection_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[JSON] Results saved → {path}")
    return path

CAPTURE_DIR = os.path.join(tempfile.gettempdir(), "vggt_guided_b3")
shutil.rmtree(CAPTURE_DIR, ignore_errors=True)
os.makedirs(CAPTURE_DIR)

# ─── Models ───────────────────────────────────────────────────────────────────
_t = time.time()
print("[VGGT] Loading...")
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
vggt_model.eval()
T_VGGT_LOAD = time.time() - _t
print(f"[VGGT] Ready.  ({T_VGGT_LOAD:.1f}s)")

_t = time.time()
print("[DINO] Loading...")
dino_processor = AutoProcessor.from_pretrained(DINO_REPO)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_REPO).to(DEVICE)
T_DINO_LOAD = time.time() - _t
print(f"[DINO] Ready.  ({T_DINO_LOAD:.1f}s)")

_t = time.time()
print("[SAM] Loading...")
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
sam_predictor = SamPredictor(sam)
T_SAM_LOAD = time.time() - _t
print(f"[SAM] Ready.  ({T_SAM_LOAD:.1f}s)")

# ─── ZED ──────────────────────────────────────────────────────────────────────
print("[ZED] Opening...")
zed = sl.Camera()
params = sl.InitParameters()
params.camera_resolution = sl.RESOLUTION.HD720
params.depth_mode        = sl.DEPTH_MODE.NONE
assert zed.open(params) == sl.ERROR_CODE.SUCCESS, "ZED failed to open"
img_mat = sl.Mat()
print("[ZED] Ready.\n")

def grab_frame():
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = img_mat.get_data()[:, :, :3]
        # ZED → BGR; convert to RGB so colours are correct for VGGT/DINO
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None

def save_frame(frame_rgb, path):
    """Center-crop to square then save as RGB PNG for VGGT.
    1280×720 → 720×720 removes the black letterbox VGGT's preprocessor would add."""
    H, W = frame_rgb.shape[:2]
    side  = min(H, W)
    y0    = (H - side) // 2
    x0    = (W - side) // 2
    crop  = frame_rgb[y0 : y0 + side, x0 : x0 + side]
    PILImage.fromarray(crop).save(path)

# ─── Pipeline ─────────────────────────────────────────────────────────────────
def run_vggt_multiview(image_paths):
    print(f"[VGGT] Preprocessing {len(image_paths)} frames...")
    images = load_and_preprocess_images(image_paths).to(DEVICE)
    print(f"[VGGT] Input shape: {tuple(images.shape)}  — inferring...")
    with torch.no_grad():
        pred = vggt_model(images)

    world_points = pred["world_points"][0].cpu().numpy()        # (S, H, W, 3)
    world_conf   = pred["world_points_conf"][0].cpu().numpy()   # (S, H, W)

    # pred["images"] is ImageNet-normalized internally — wrong colors for point cloud.
    # Load original saved PNGs and resize to VGGT's output spatial resolution instead.
    S, H_out, W_out = world_points.shape[:3]
    rgb_frames = np.zeros((S, H_out, W_out, 3), dtype=np.float32)
    for i, p in enumerate(image_paths):
        img = PILImage.open(p).resize((W_out, H_out), PILImage.LANCZOS)
        rgb_frames[i] = np.asarray(img, dtype=np.float32) / 255.0

    return world_points, world_conf, rgb_frames

def run_dino(pil_rgb):
    """pil_rgb: PIL.Image in RGB mode (original frame, not VGGT-normalized)."""
    prompt = ". ".join(c.lower().strip() for c in QUERY_CLASSES) + "."
    inputs = dino_processor(images=pil_rgb, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    target_sizes = torch.tensor([pil_rgb.size[::-1]]).to(DEVICE)
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=DINO_BOX_THR, text_threshold=DINO_TEXT_THR,
        target_sizes=target_sizes,
    )[0]
    return [{"bbox": b.cpu().numpy().tolist(), "confidence": float(s), "label": l}
            for b, s, l in zip(results["boxes"], results["scores"], results["text_labels"])]

def nms_detections(dets, iou_thr=0.50):
    if not dets:
        return dets
    boxes  = np.array([d["bbox"] for d in dets], dtype=np.float32)
    scores = np.array([d["confidence"] for d in dets])
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        x1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        y1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        x2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        y2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        area_i    = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_rest = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou   = inter / (area_i + area_rest - inter + 1e-6)
        order = order[1:][iou < iou_thr]
    return [dets[i] for i in keep]

def label_matches(query, dino_label):
    for qw in query.lower().strip().split():
        for lw in dino_label.lower().strip().split():
            if qw == lw:
                return True
            shorter, longer = (qw, lw) if len(qw) <= len(lw) else (lw, qw)
            if len(shorter) >= 5 and longer.startswith(shorter):
                return True
    return False

def run_sam(pil_rgb, bbox):
    """pil_rgb: PIL.Image in RGB mode."""
    rgb_u8 = np.asarray(pil_rgb, dtype=np.uint8)
    sam_predictor.set_image(rgb_u8)
    masks, scores, _ = sam_predictor.predict(
        box=np.array(bbox, dtype=np.float32), multimask_output=False)
    mask = masks[0].astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lbl_cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbl_cc == largest).astype(np.float32)
    return mask, float(scores[0])

def build_scene(world_points, conf, rgb_vggt, labeled_masks):
    S, H, W, _ = world_points.shape
    pts_all  = world_points.reshape(-1, 3)
    cols_all = rgb_vggt.reshape(-1, 3).astype(np.float32)
    conf_all = conf.reshape(-1)

    fin = np.isfinite(conf_all)
    conf_thr = np.percentile(conf_all[fin], 5)    # keep top 95%
    dist     = np.linalg.norm(pts_all, axis=-1)
    valid = np.isfinite(pts_all).all(axis=1) & (conf_all > conf_thr) & (dist < 10.0)
    print(f"[BUILD] conf thr={conf_thr:.3f}  "
          f"(range {conf_all[fin].min():.3f}–{conf_all[fin].max():.3f})  "
          f"valid={valid.sum():,}")

    pts  = pts_all[valid]
    cols = cols_all[valid].copy()

    per_label_flat = {}
    obj_any_flat   = np.zeros(S * H * W, dtype=bool)
    for label, (mask, frame_idx) in labeled_masks.items():
        arr   = np.zeros(S * H * W, dtype=bool)
        start = frame_idx * H * W
        arr[start : start + H * W] = (mask.reshape(-1) > 0.5)
        obj_any_flat |= arr
        per_label_flat[label] = arr[valid]

    obj_any_valid = obj_any_flat[valid]

    if len(pts) > MAX_SCENE_PTS:
        idx  = np.random.choice(len(pts), MAX_SCENE_PTS, replace=False)
        keep = np.zeros(len(pts), dtype=bool)
        keep[idx] = True
        keep |= obj_any_valid
        pts  = pts[keep]
        cols = cols[keep]
        per_label_flat = {l: f[keep] for l, f in per_label_flat.items()}

    for label, flat in per_label_flat.items():
        color = CLASS_COLORS.get(label.lower().strip(), DEFAULT_COLOR)
        cols[flat] = color

    print(f"[BUILD] {len(pts):,} pts before cleanup")
    if len(pts) >= 100:
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(pts)
        tmp.colors = o3d.utility.Vector3dVector(cols)
        tmp = tmp.voxel_down_sample(voxel_size=VOXEL_SIZE)
        tmp, _ = tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
        if len(tmp.points) > 0:
            pts  = np.asarray(tmp.points).astype(np.float32)
            cols = np.asarray(tmp.colors).astype(np.float32)
    print(f"[BUILD] {len(pts):,} pts after cleanup")
    return pts.astype(np.float32), cols.astype(np.float32)

# ─── Open3D window ────────────────────────────────────────────────────────────
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("VGGT Active Perception [basetest3]", 1280, 720)
vis.add_geometry(pcd)
vis.get_render_option().point_size       = 2.5
vis.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
vis.get_render_option().light_on         = False

# ─── Phase 1: Auto-capture 60 frames then proceed ────────────────────────────
captured_paths  = []
capture_times   = []
last_capture_t  = 0.0
T_CAPTURE_START = None

print(f"Keep sweeping LEFT ↔ RIGHT — capturing {MAX_FRAMES} frames automatically. Q = quit.\n")

try:
    while len(captured_paths) < MAX_FRAMES:
        frame = grab_frame()
        if frame is None:
            continue

        now   = time.time()
        n_cap = len(captured_paths)

        # Auto-capture every CAPTURE_INTERVAL seconds — no button needed
        if (now - last_capture_t) >= CAPTURE_INTERVAL:
            path = os.path.join(CAPTURE_DIR, f"frame_{n_cap:02d}.png")
            save_frame(frame, path)
            captured_paths.append(path)
            capture_times.append(now)
            last_capture_t = now
            if T_CAPTURE_START is None:
                T_CAPTURE_START = now
            print(f"[CAPTURE] {n_cap + 1}/{MAX_FRAMES}  [+{now - T_GOAL:.1f}s]")

        # ── Draw UI ───────────────────────────────────────────────────────────
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        disp      = cv2.resize(frame_bgr, (640, 360))
        H, W      = disp.shape[:2]

        # Header
        bar = disp.copy()
        cv2.rectangle(bar, (0, 0), (W, 56), (10, 10, 14), -1)
        cv2.addWeighted(bar, 0.80, disp, 0.20, 0, disp)
        cv2.circle(disp, (W - 22, 28), 9, (0, 0, 220), -1)   # red rec dot
        cv2.putText(disp, f"Capturing  {n_cap + 1} / {MAX_FRAMES}",
                    (14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.95, (50, 230, 80), 2, cv2.LINE_AA)

        # Progress bar
        filled = int((n_cap / MAX_FRAMES) * (W - 4))
        cv2.rectangle(disp, (2, H - 16), (W - 2, H - 2), (35, 35, 35), -1)
        if filled > 0:
            cv2.rectangle(disp, (2, H - 16), (2 + filled, H - 2), (50, 210, 80), -1)
        cv2.putText(disp, "Keep sweeping LEFT  <->  RIGHT",
                    (14, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("ZED — Auto Capture", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit(0)

        if not vis.poll_events():
            raise SystemExit(0)
        vis.update_renderer()

    # ── ADAPT: sweep done → precondition satisfied ────────────────────────────
    ADAPT_GOAL["preconditions"]["sweep_complete"] = True
    print(f"\n[ADAPT] sweep_complete=True — investigate-discrepancy operator firing")

    # ─── Phase 2: Processing screen ───────────────────────────────────────────
    n_cap = len(captured_paths)
    proc  = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(proc, f"All {n_cap} frames captured!",
                (30, 100), cv2.FONT_HERSHEY_DUPLEX, 0.95, (50, 230, 80), 2, cv2.LINE_AA)
    cv2.putText(proc, "Running VGGT + DINO + SAM...",
                (30, 160), cv2.FONT_HERSHEY_DUPLEX, 0.85, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(proc, "(please wait, this takes a few minutes)",
                (30, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.imshow("ZED — Capture", proc)
    cv2.waitKey(1)

    # ─── Subsample to VGGT_MAX_FRAMES evenly-spaced frames ───────────────────
    if len(captured_paths) > VGGT_MAX_FRAMES:
        indices = np.linspace(0, len(captured_paths) - 1, VGGT_MAX_FRAMES, dtype=int).tolist()
        vggt_paths = [captured_paths[i] for i in indices]
        print(f"[SUBSAMPLE] {len(captured_paths)} captured → {len(vggt_paths)} evenly-spaced for VGGT")
    else:
        vggt_paths = captured_paths

    # ─── Phase 3: VGGT inference ──────────────────────────────────────────────
    T_INFER_START = time.time()
    try:
        world_points, conf, rgb_vggt = run_vggt_multiview(vggt_paths)
        ADAPT_GOAL["preconditions"]["reconstruction_ready"] = True
    except Exception as e:
        import traceback
        print(f"[ERROR] VGGT failed: {e}\n{traceback.format_exc()}")
        raise SystemExit(1)
    T_VGGT_INFER = time.time() - T_INFER_START

    print(f"[VGGT] world_points: {world_points.shape}  conf: {conf.shape}")
    print(f"[VGGT] conf  min={conf.min():.3f}  max={conf.max():.3f}  "
          f"mean={conf.mean():.3f}  25th-pct={np.percentile(conf, 25):.3f}")

    # ─── Phase 4: DINO + SAM ──────────────────────────────────────────────────
    # Load original frames as PIL (correct RGB, not VGGT-normalized)
    pil_frames = [PILImage.open(p).convert("RGB") for p in vggt_paths]

    labeled_masks = {}
    T_DINO_INFER  = 0.0
    T_SAM_INFER   = 0.0
    S_frames      = len(pil_frames)

    # world_points spatial dims — masks need to be resized to this for projection
    H_wp, W_wp = rgb_vggt.shape[1], rgb_vggt.shape[2]

    detection_records = []
    not_found_classes = []

    try:
        print(f"\n[ADAPT] Operator: investigate-discrepancy — running DINO+SAM on focal ROI")
        dino_indices = np.linspace(0, S_frames - 1, min(5, S_frames), dtype=int).tolist()
        print(f"[DINO] scanning frames {[i+1 for i in dino_indices]} of {S_frames}  "
              f"target={QUERY_CLASSES}")
        all_dets = []
        for fi in dino_indices:
            _t   = time.time()
            dets = run_dino(pil_frames[fi])
            T_DINO_INFER += time.time() - _t
            for d in dets:
                d["frame_idx"] = fi
            matched = [d for d in dets if any(label_matches(c, d["label"]) for c in QUERY_CLASSES)]
            all_dets.extend(matched)
            if matched:
                ADAPT_GOAL["preconditions"]["focal_roi_set"] = True
            print(f"  frame {fi+1:>2}: {len(matched)} matched  ({T_DINO_INFER:.1f}s total)")

        for cls in QUERY_CLASSES:
            cls_dets = [d for d in all_dets if label_matches(cls, d["label"])]
            if not cls_dets:
                print(f"[WARN] '{cls}' not found in any frame.")
                not_found_classes.append(cls)
                continue

            anchor    = max(cls_dets, key=lambda d: d["confidence"])
            fi_anchor = anchor["frame_idx"]
            all_confs = sorted([d["confidence"] for d in cls_dets], reverse=True)
            print(f"[DINO] '{cls}' anchor=frame {fi_anchor+1}  conf={anchor['confidence']:.3f}")

            anchor_dets = nms_detections(
                [d for d in cls_dets if d["frame_idx"] == fi_anchor])[:MAX_SAM_PER_CLASS]
            print(f"  after NMS: {len(anchor_dets)} boxes → SAM")

            _t        = time.time()
            pil_anchor = pil_frames[fi_anchor]
            combined_mask = np.zeros((pil_anchor.height, pil_anchor.width), dtype=np.float32)
            sam_scores = []
            for i, d in enumerate(anchor_dets):
                mask, sam_score = run_sam(pil_anchor, d["bbox"])
                combined_mask = np.maximum(combined_mask, mask)
                sam_scores.append(round(float(sam_score), 4))
                print(f"  [{i+1}/{len(anchor_dets)}] conf={d['confidence']:.3f}  "
                      f"mask={int(mask.sum()):,}px  sam={sam_score:.3f}")
            T_SAM_INFER += time.time() - _t

            mask_wp = cv2.resize(combined_mask, (W_wp, H_wp), interpolation=cv2.INTER_LINEAR)
            print(f"[SAM] '{cls}' mask={int(combined_mask.sum()):,}px → "
                  f"resized {W_wp}×{H_wp}  ({T_SAM_INFER:.1f}s)")
            labeled_masks[cls] = (mask_wp, fi_anchor)

            # ── ADAPT: port_state + 3D pose + RS schema ───────────────────────
            best_conf  = float(anchor["confidence"])
            port_state = compute_port_state(best_conf)
            pose_xyz   = extract_pose_xyz(world_points, mask_wp, fi_anchor)
            action, fb_msg = cognitive_feedback(port_state, cls)

            print(f"[ADAPT] '{cls}'  conf={best_conf:.3f}  port_state={port_state}")
            print(fb_msg)
            if pose_xyz:
                print(f"  pose   : x={pose_xyz['x']:.3f}  y={pose_xyz['y']:.3f}  z={pose_xyz['z']:.3f} (m)")

            rgb = CLASS_COLORS.get(cls.lower(), DEFAULT_COLOR)
            anchor_bbox = anchor["bbox"]
            detection_records.append({
                # ── RS schema fields (Table 1 §5.2) ──────────────────────────
                "schema_type":    "RS",
                "persistent_id":  uuid.uuid4().hex[:10],
                "class":          cls,
                "port_state":     port_state,
                "image_region": {
                    "x1": round(anchor_bbox[0], 1), "y1": round(anchor_bbox[1], 1),
                    "x2": round(anchor_bbox[2], 1), "y2": round(anchor_bbox[3], 1),
                    "frame": int(fi_anchor) + 1,
                },
                "geometry":       "3D_point_cloud_VGGT",
                "pose":           pose_xyz,
                "confidence":     round(best_conf, 4),
                "next_action":    action,
                # ── visualisation / book-keeping ──────────────────────────────
                "colour":          colour_name(cls),
                "colour_rgb":      [round(float(v), 3) for v in rgb],
                "all_confidences": [round(float(c), 4) for c in all_confs],
                "detections_found": len(cls_dets),
                "boxes_after_nms":  len(anchor_dets),
                "sam_scores":       sam_scores,
                "anchor_frame":     int(fi_anchor) + 1,
            })

        # ── ADAPT cognitive summary ───────────────────────────────────────────
        if detection_records:
            print(f"\n[ADAPT] ══ COGNITIVE FEEDBACK SUMMARY ══")
            for rec in detection_records:
                print(f"  {rec['class']:<20} port_state={rec['port_state']:<12} "
                      f"conf={rec['confidence']:.3f}  action={rec['next_action']}")
            if not_found_classes:
                print(f"  NOT FOUND: {not_found_classes}")

    except Exception as e:
        import traceback
        print(f"[WARN] DINO/SAM failed ({e}) — showing raw reconstruction\n{traceback.format_exc()}")

    # ─── Phase 5: Build scene ─────────────────────────────────────────────────
    _t = time.time()
    pts, cols = build_scene(world_points, conf, rgb_vggt, labeled_masks)
    T_BUILD = time.time() - _t

    if len(pts) == 0:
        print("[WARN] 0 pts after filter — using raw fallback")
        flat_pts  = world_points.reshape(-1, 3)
        flat_cols = rgb_vggt.reshape(-1, 3).astype(np.float32)
        valid     = np.isfinite(flat_pts).all(axis=1)
        pts, cols = flat_pts[valid], flat_cols[valid]

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    vis.update_geometry(pcd)
    vis.reset_view_point(True)

    # Set a 45° elevated view so depth is visible (not flat straight-on)
    ctr = vis.get_view_control()
    ctr.rotate(0, -300)   # tilt up to show depth
    ctr.rotate(200, 0)    # slight horizontal rotation

    T_DONE     = time.time()
    T_CAP_SPAN = (capture_times[-1] - capture_times[0]) if len(capture_times) > 1 else 0.0
    T_INFER    = T_VGGT_INFER + T_DINO_INFER + T_SAM_INFER + T_BUILD
    T_TOTAL    = T_DONE - T_GOAL

    W = 52
    sep  = "─" * W
    dsep = "═" * W
    def row(label, val, note=""):
        s = f"  {label:<28} {val:>7}"
        if note: s += f"   {note}"
        print(s)

    print(f"\n╔{dsep}╗")
    print(f"║{'  PIPELINE TIMING SUMMARY':^{W}}║")
    print(f"╠{dsep}╣")
    print(f"║  {'STEP':<28} {'TIME':>7}   {'NOTE':<10}║")
    print(f"╠{sep}╣")
    print(f"║  {'[LOAD] VGGT':<28} {T_VGGT_LOAD:>6.1f}s   one-time       ║")
    print(f"║  {'[LOAD] DINO':<28} {T_DINO_LOAD:>6.1f}s   one-time       ║")
    print(f"║  {'[LOAD] SAM':<28} {T_SAM_LOAD:>6.1f}s   one-time       ║")
    print(f"╠{sep}╣")
    print(f"║  {'[CAPTURE] ZED sweep':<28} {T_CAP_SPAN:>6.1f}s   {len(captured_paths)} frames        ║")
    print(f"║  {'[SUBSAMPLE] 60→9 frames':<28} {'<0.1':>6}s                  ║")
    print(f"╠{sep}╣")
    print(f"║  {'[VGGT] 3D inference':<28} {T_VGGT_INFER:>6.1f}s   9 frames        ║")
    print(f"║  {'[DINO] object detection':<28} {T_DINO_INFER:>6.1f}s   5 frames scanned║")
    print(f"║  {'[SAM] segmentation':<28} {T_SAM_INFER:>6.1f}s                  ║")
    print(f"║  {'[BUILD] scene + cleanup':<28} {T_BUILD:>6.1f}s   voxel+outlier   ║")
    print(f"╠{sep}╣")
    print(f"║  {'Inference subtotal':<28} {T_INFER:>6.1f}s                  ║")
    print(f"╠{dsep}╣")
    print(f"║  {'TOTAL  (goal → Open3D)':<28} {T_TOTAL:>6.1f}s   ({T_TOTAL/60:.1f} min)      ║")
    print(f"╚{dsep}╝")

    # ─── Save detection results to JSON ───────────────────────────────────────
    save_results_json(
        detection_records = detection_records,
        not_found         = not_found_classes,
        timing            = {
            "load_vggt_s":      round(T_VGGT_LOAD, 1),
            "load_dino_s":      round(T_DINO_LOAD, 1),
            "load_sam_s":       round(T_SAM_LOAD, 1),
            "capture_span_s":   round(T_CAP_SPAN, 1),
            "frames_captured":  len(captured_paths),
            "frames_to_vggt":   len(vggt_paths),
            "vggt_infer_s":     round(T_VGGT_INFER, 1),
            "dino_infer_s":     round(T_DINO_INFER, 1),
            "sam_infer_s":      round(T_SAM_INFER, 1),
            "scene_build_s":    round(T_BUILD, 1),
            "inference_total_s": round(T_INFER, 1),
            "total_s":          round(T_TOTAL, 1),
            "total_min":        round(T_TOTAL / 60, 2),
        },
    )

    print("\nDone. Explore in Open3D. Close window to exit.")

    # ─── Phase 6: View loop ───────────────────────────────────────────────────
    cv2.destroyAllWindows()
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

except SystemExit:
    pass
finally:
    zed.close()
    cv2.destroyAllWindows()
    vis.destroy_window()
