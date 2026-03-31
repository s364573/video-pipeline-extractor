from pathlib import Path
import subprocess
import pandas as pd
from feat import Detector
import torch
from utils.config_loader import load_config, get_device
DEVICE = get_device()
_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]
import os
os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ["PATH"]
# -------------------------
# CONFIG
# -------------------------
TARGET_WIDTH = 640
SKIP_FRAMES = 5
BATCH_SIZE = 64

CONFIG_TAG = f"pyfeat_w{TARGET_WIDTH}_fps50_skip{SKIP_FRAMES}"
CACHE_TAG = f"downscaled_w{TARGET_WIDTH}"

_detector = None


# -------------------------
# DETECTOR
# -------------------------
def get_detector():
    global _detector
    if _detector is None:
        print("[INIT] Loading py-feat detector...")
        _detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            device=DEVICE
        )
        print(f"[INIT] Device: {DEVICE}")
        print(f"[INIT] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    return _detector


# -------------------------
# DOWNSCALE (CACHED)
# -------------------------
def downscale(video_path: Path, base_path: Path) -> Path:
    rel = video_path.relative_to(base_path)

    cache_path = base_path / "cache" / CACHE_TAG / rel
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path

    cmd = [
        FFMPEG,
        "-y",
        "-i", str(video_path),
        "-vf", f"scale={TARGET_WIDTH}:-1",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(cache_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cache_path


# -------------------------
# PROCESS ONE CLIP (FIXED)
# -------------------------
def process_clip(video_path: Path, base_path: Path):
    detector = get_detector()

    video_ds = downscale(video_path, base_path)

    rel = video_path.relative_to(base_path)

    output_csv = (
        base_path
        / "features"
        / CONFIG_TAG
        / rel.with_suffix(".csv")
    )

    if output_csv.exists():
        print(f"[SKIP] {output_csv.name}")
        return

    print(f"[FEAT] Processing {video_path.name}")

    try:
        result = detector.detect_video(str(video_ds),batch_size=BATCH_SIZE, skip_frames=SKIP_FRAMES)  # ← let py-feat handle skipping)
    except Exception as e:
        print(f"[ERROR] {video_path.name}: {e}")
        return

    if result is None or len(result) == 0:
        print(f"[WARN] No features: {video_path.name}")
        return

    df = pd.DataFrame(result)

    # -------------------------
    # ADD TIME / FRAME
    # -------------------------
    # py-feat index = frame index
    df["frame"] = df.index

    # estimate fps (py-feat does not always expose it reliably)
    # fallback: assume 50 fps input
    fps = 50
    df["time"] = df["frame"] / fps
    

    df["config"] = CONFIG_TAG

    # -------------------------
    # SAVE
    # -------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[OK] {output_csv}")


# -------------------------
# MAIN ENTRY
# -------------------------
def extract_features(base_path: Path, clip_types=None):
    """
    clip_types: list of folders to process, in order.
    Options: ["stimuli"], ["questions"], ["stimuli", "questions"]
    Default: stimuli only (most important for AU extraction)
    """
    if clip_types is None:
        clip_types = ["stimuli"]  # default: stimuli only

    print("[FEATURES] Starting extraction...")

    videos = []
    for clip_type in clip_types:
        folder = base_path / clip_type
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue
        found = list(folder.glob("*.mp4"))
        found = [v for v in found if "cache" not in str(v)]
        print(f"[INFO] {clip_type}: {len(found)} clips")
        videos.extend(sorted(found))

    print(f"[FEATURES] Total: {len(videos)} clips")
    for v in videos:
        process_clip(v, base_path)
    print("[FEATURES] Done.")