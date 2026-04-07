import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import pandas as pd
from feat import Detector
from utils.config_loader import load_config, get_device

DEVICE = get_device()
_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]
os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ["PATH"]

# -------------------------
# CONFIG
# -------------------------
TARGET_WIDTH = 640
SKIP_FRAMES  = 5

# GPU: single process, large batches — GPU handles parallelism internally
# CPU: multiprocessing pool, smaller batches
if DEVICE == "cuda":
    NUM_WORKERS       = 1
    BATCH_SIZE        = 256
else:
    NUM_WORKERS       = min(3, multiprocessing.cpu_count())
    BATCH_SIZE        = 64

DOWNSCALE_WORKERS = 4   # parallel FFmpeg — safe, each clip has a unique output path

CONFIG_TAG = f"pyfeat_w{TARGET_WIDTH}_fps50_skip{SKIP_FRAMES}"
CACHE_TAG  = f"downscaled_w{TARGET_WIDTH}"


# -------------------------
# DETECTOR
# -------------------------
_detector = None

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
        FFMPEG, "-y",
        "-i", str(video_path),
        "-vf", f"scale={TARGET_WIDTH}:-1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-an",
        str(cache_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cache_path


def _downscale_task(args):
    return downscale(*args)


# -------------------------
# CORE CLIP PROCESSING
# -------------------------
def process_clip(video_path: Path, base_path: Path, detector=None):
    if detector is None:
        detector = get_detector()

    video_ds = downscale(video_path, base_path)
    rel = video_path.relative_to(base_path)
    output_csv = base_path / "features" / CONFIG_TAG / rel.with_suffix(".csv")

    if output_csv.exists():
        print(f"[SKIP] {video_path.name}")
        return

    print(f"[FEAT] {video_path.name}")

    try:
        result = detector.detect_video(str(video_ds), batch_size=BATCH_SIZE, skip_frames=SKIP_FRAMES)
    except Exception as e:
        print(f"[ERROR] {video_path.name}: {e}")
        return

    if result is None or len(result) == 0:
        print(f"[WARN] No features: {video_path.name}")
        return

    df = pd.DataFrame(result)
    df["frame"] = df.index
    df["time"]  = df["frame"] / 50
    df["config"] = CONFIG_TAG

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] {output_csv.name}")


# -------------------------
# MULTIPROCESSING WORKER (CPU only)
# -------------------------
_worker_detector = None

def _init_worker():
    global _worker_detector
    _worker_detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        device="cpu"
    )


def _process_clip_worker(args):
    video_path_str, base_path_str = args
    process_clip(Path(video_path_str), Path(base_path_str), detector=_worker_detector)


# -------------------------
# MAIN ENTRY
# -------------------------
def extract_features(base_path: Path, clip_types=None):
    if clip_types is None:
        clip_types = ["stimuli"]

    print("[FEATURES] Starting extraction...")

    videos = []
    for clip_type in clip_types:
        folder = base_path / clip_type
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue
        found = sorted(v for v in folder.glob("*.mp4") if "cache" not in str(v))
        print(f"[INFO] {clip_type}: {len(found)} clips")
        videos.extend(found)

    if not videos:
        print("[FEATURES] Nothing to do.")
        return

    print(f"[FEATURES] {len(videos)} clips | device: {DEVICE} | workers: {NUM_WORKERS} | batch: {BATCH_SIZE}")

    # Pre-load detector in main process to ensure model files are cached
    # before any worker processes start (prevents download race condition)
    get_detector()

    # Downscale all videos in parallel — safe because each clip has a unique output path
    print(f"[FEATURES] Downscaling ({DOWNSCALE_WORKERS} parallel FFmpeg)...")
    with ThreadPoolExecutor(max_workers=DOWNSCALE_WORKERS) as executor:
        list(executor.map(_downscale_task, [(v, base_path) for v in videos]))

    # Inference — strategy depends on device
    if DEVICE == "cuda":
        # Single process on GPU: more efficient than spawning multiple CUDA processes
        detector = get_detector()
        for v in videos:
            process_clip(v, base_path, detector=detector)
    else:
        # CPU/MPS: multiprocessing pool, each worker loads its own detector
        args = [(str(v), str(base_path)) for v in videos]
        with multiprocessing.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
            pool.map(_process_clip_worker, args)

    print("[FEATURES] Done.")
