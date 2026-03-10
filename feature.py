from pathlib import Path
import subprocess
import torch
from feat import Detector

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    device=device,
)


def compress_video(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",          # hide ffmpeg spam
        "-i", str(input_path),
        "-vf", "fps=8,scale=960:-1",   # downsample
        "-c:v", "libx264",
        "-preset", "fast",
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video(video_path, feat_dir):

    tmp = video_path.with_name(video_path.stem + ".small.mp4")

    # Remove corrupted temp file if it exists
    if tmp.exists():
        tmp.unlink()

    compress_video(video_path, tmp)

    predictions = detector.detect_video(
        str(tmp),
        batch_size=32
    )

    output_file = feat_dir / f"{video_path.stem}.csv"
    predictions.to_csv(output_file)

    if tmp.exists():
        tmp.unlink()


def extract_features(pid, block):

    block_dir = Path("participant") / pid / f"B{block}"
    feat_dir = block_dir / "features"
    feat_dir.mkdir(exist_ok=True)

    # ignore temp files
    videos = sorted(
        v for v in block_dir.glob("*.mp4")
        if ".small" not in v.name
    )

    print(f"Found {len(videos)} videos")

    for v in videos:

        output_file = feat_dir / f"{v.stem}.csv"

        if output_file.exists():
            print("Skipping", v.name)
            continue

        print("Processing", v.name)

        try:
            process_video(v, feat_dir)

        except Exception as e:
            print("Error processing", v.name, ":", e)


if __name__ == "__main__":
    extract_features("RRRR", "00")