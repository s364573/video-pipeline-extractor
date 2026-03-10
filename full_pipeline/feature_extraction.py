from pathlib import Path
import subprocess
import torch
import argparse
from feat import Detector

ROOT = Path(__file__).resolve().parent.parent
# ---------------------------------
# QUALITY PRESETS
# ---------------------------------

QUALITY_PRESETS = {
    "4k_50fps": {"fps": 50, "scale": 4096},
    "2k_25fps": {"fps": 25, "scale": 2048},
    "1k_16fps": {"fps": 16, "scale": 1024},
    "960_8fps": {"fps": 8, "scale": 960},
    "640_4fps": {"fps": 4, "scale": 640},
}


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

def compress_video(input_path, output_path, fps, scale):

    vf = f"fps={fps},scale={scale}:-1"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", vf,
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

def process_video(video_path, feat_dir, fps, scale, batch_size):

    tmp = video_path.with_suffix(".small.mp4")

    try:

        compress_video(video_path, tmp, fps, scale)

        predictions = detector.detect_video(
            str(tmp),
            batch_size=batch_size
        )

        predictions.to_csv(feat_dir / f"{video_path.stem}.csv")

    finally:

        if tmp.exists():
            tmp.unlink()


# ---------------------------------
# PROCESS BLOCK
# ---------------------------------

def extract_features(pid, block, quality, batch_size):
    

    settings = QUALITY_PRESETS[quality]

    fps = settings["fps"]
    scale = settings["scale"]

    block_dir = ROOT / "participant" / "processed" / pid / f"B{block}"

    feat_dir = block_dir / "features" / quality
    feat_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        v for v in block_dir.glob("*.mp4")
        if ".small" not in v.stem
    )

    print(f"Found {len(videos)} videos")
    print(f"Quality: {quality} (fps={fps}, scale={scale})")

    for v in videos:

        output = feat_dir / f"{v.stem}.csv"

        if output.exists():
            print("Skipping", v.name)
            continue

        print("Processing", v.name)

        try:
            process_video(v, feat_dir, fps, scale, batch_size)

        except Exception as e:
            print("Error processing", v.name, ":", e)


# ---------------------------------
# CLI
# ---------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("participant")
    parser.add_argument("block")

    parser.add_argument(
        "quality",
        choices=QUALITY_PRESETS.keys(),
        help="processing quality"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="batch size for detector"
    )

    args = parser.parse_args()

    extract_features(
        args.participant,
        args.block,
        args.quality,
        args.batch
    )


# ---------------------------------

if __name__ == "__main__":
    main()