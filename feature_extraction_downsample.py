from pathlib import Path
import subprocess
import torch
from feat import Detector


# ---------------------------------
# DEVICE
# ---------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")


# ---------------------------------
# LOAD DETECTOR
# ---------------------------------

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    device=device,
)


# ---------------------------------
# VIDEO COMPRESSION
# ---------------------------------

def compress_video(input_path, output_path):

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),

        # downsample + resize
        "-vf", "fps=8,scale=960:-1",

        # remove audio (faster)
        "-an",

        "-c:v", "libx264",
        "-preset", "fast",

        str(output_path)
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# ---------------------------------
# PROCESS ONE VIDEO
# ---------------------------------

def process_video(video_path, feat_dir):

    tmp = video_path.with_suffix(".small.mp4")

    try:

        compress_video(video_path, tmp)

        predictions = detector.detect_video(
            str(tmp),
            batch_size=64
        )

        predictions.to_csv(feat_dir / f"{video_path.stem}.csv")

    finally:

        if tmp.exists():
            tmp.unlink()


# ---------------------------------
# PROCESS BLOCK
# ---------------------------------

def extract_features(pid, block):

    block_dir = Path("participant") / pid / f"B{block}"
    feat_dir = block_dir / "features"

    feat_dir.mkdir(exist_ok=True)

    # Only process original videos
    videos = sorted(
        v for v in block_dir.glob("*.mp4")
        if ".small" not in v.stem
    )

    print(f"Found {len(videos)} videos")

    for v in videos:

        output = feat_dir / f"{v.stem}.csv"

        if output.exists():
            print("Skipping", v.name)
            continue

        print("Processing", v.name)

        try:
            process_video(v, feat_dir)

        except Exception as e:
            print("Error processing", v.name, ":", e)


# ---------------------------------
# RUN
# ---------------------------------

if __name__ == "__main__":

    extract_features("RRRR", "03")