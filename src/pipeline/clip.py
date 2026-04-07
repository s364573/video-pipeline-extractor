import subprocess
from pathlib import Path
from utils.config_loader import load_config

_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]


def clip_video(input_path: Path, start: float, end: float, output_path: Path):
    duration = end - start

    if duration <= 0:
        raise ValueError(f"Invalid duration: {start} → {end}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -ss before -i = fast input seeking (avoids decoding from start)
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

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {output_path}")


def clip_trials(block, labels, output_root):
    if not Path(block.video_path).exists():
        raise FileNotFoundError(f"Video not found: {block.video_path}")

    for clip in labels:
        out_path = output_root / block.participant_id / block.block_id / clip["file"]
        clip_video(block.video_path, clip["start"], clip["end"], out_path)
        print(f"[CLIP] {clip['type']} → {out_path.name}")