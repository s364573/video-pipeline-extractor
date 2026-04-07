import sys
from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).parent.parent  # src/utils/ → src/


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config():
    config_path = PROJECT_ROOT / "config.json"
    assert config_path.exists(), f"config.json not found at {config_path}"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "raw_root":     Path(cfg["raw_root"]),
        "output_root":  Path(cfg["output_root"]),
        "project_root": PROJECT_ROOT,
        "ffmpeg":       cfg.get("ffmpeg", "ffmpeg"),
        "venv_feat":    cfg.get("venv_feat", "../venv_feat"),
    }


def get_feat_python(cfg=None):
    """
    Returns the path to the Python executable inside venv_feat.
    Handles Windows (Scripts/python.exe) vs Mac/Linux (bin/python).
    venv_feat path in config.json is relative to the project root (src/).
    """
    if cfg is None:
        cfg = load_config()

    venv_feat_raw = cfg["venv_feat"]

    # Resolve relative paths from project root
    venv_feat = Path(venv_feat_raw)
    if not venv_feat.is_absolute():
        venv_feat = (cfg["project_root"] / venv_feat_raw).resolve()

    if sys.platform == "win32":
        return venv_feat / "Scripts" / "python.exe"
    else:
        return venv_feat / "bin" / "python"
