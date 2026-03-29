from pathlib import Path
import subprocess
import pandas as pd
from feat import Detector

# -------------------------
# CONFIG
# -------------------------
TARGET_WIDTH = 640
SKIP_FRAMES = 2

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
            au_model="xgb"
        )
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
        "ffmpeg",
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
        result = detector.detect_video(str(video_ds))
    except Exception as e:
        print(f"[ERROR] {video_path.name}: {e}")
        return

    if result is None or len(result) == 0:
        print(f"[WARN] No features: {video_path.name}")
        return

    df = result.to_pandas()

    # -------------------------
    # ADD TIME / FRAME
    # -------------------------
    # py-feat index = frame index
    df["frame"] = df.index

    # estimate fps (py-feat does not always expose it reliably)
    # fallback: assume 50 fps input
    fps = 50
    df["time"] = df["frame"] / fps

    # -------------------------
    # APPLY FRAME SKIP (AFTER)
    # -------------------------
    if SKIP_FRAMES > 1:
        df = df.iloc[::SKIP_FRAMES].copy()

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
def extract_features(base_path: Path):
    print("[FEATURES] Starting extraction...")

    videos = list(base_path.rglob("*.mp4"))
    print(f"[FEATURES] Found {len(videos)} clips")

    for v in videos:
        process_clip(v, base_path)

    print("[FEATURES] Done.")