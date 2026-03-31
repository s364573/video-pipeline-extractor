# run_features.py
# Called by main.py as subprocess:
# venv_feat\Scripts\python.exe run_features.py <participant_id> <block_id>
import sys
from pathlib import Path
from utils.config_loader import load_config
from pipeline.features import extract_features

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: run_features.py <participant_id> <block_id>"

    participant_id = sys.argv[1]
    block_id = sys.argv[2]

    cfg = load_config()
    base_path = cfg["output_root"] / participant_id / block_id

    print(f"[feat] base_path: {base_path}")
    assert base_path.exists(), f"base_path does not exist: {base_path}"

    extract_features(base_path)
    print("[feat] Done.")