from pathlib import Path
import json
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

PROJECT_ROOT = Path(__file__).parent.parent  # src/utils/ → src/

def load_config():
    config_path = PROJECT_ROOT / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "raw_root":     Path(cfg["raw_root"]),
        "output_root":  Path(cfg["output_root"]),
        "project_root": PROJECT_ROOT,
        "ffmpeg":       cfg.get("ffmpeg", "ffmpeg")
    }