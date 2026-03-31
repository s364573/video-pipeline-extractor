import subprocess
from pathlib import Path
from utils.config_loader import load_config
_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]

def clip_video(input_path: Path, start: float, end: float, output_path: Path):
    duration = end - start

    # --- SAFETY ---
    if duration <= 0:
        raise ValueError(f"Invalid duration: {start} → {end}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c:v", "copy",
        "-c:a", "copy",
        str(output_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {output_path}")


def clip_trials(block, labels, output_root):
    participant = block.participant_id
    block_id = block.block_id
    video_path = block.video_path

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    for clip in labels:
        start = clip["start"]
        end = clip["end"]

        out_path = output_root / participant / block_id / clip["file"]

        clip_video(video_path, start, end, out_path)

        print(f"[CLIP] {clip['type']} → {out_path.name}")