"""
export_novid.py
---------------
Copies only the analysis outputs (JSON + feature CSVs) to a lightweight
export folder — no video files. Use this to create a small package you
can download/sync for analysis without the large .mp4 files.

Output structure mirrors processed/ but with no .mp4 files:
    export_novid/
        AAAD/
            B01/
                labels.json
                trials.json
                metadata.json
                sync.json
                features/
                    pyfeat_w640_fps50_skip5/
                        stimuli/
                            B01_01_surprise_stim.csv
                            ...

Usage:
    python export_novid.py                   # exports all processed participants
    python export_novid.py AAAD              # exports one participant
    python export_novid.py AAAD B01          # exports one block
"""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config

EXPORT_FILES = ["labels.json", "trials.json", "metadata.json", "sync.json"]


def export_block(processed_root: Path, export_root: Path, participant_id: str, block_id: str):
    src = processed_root / participant_id / block_id
    dst = export_root / participant_id / block_id

    if not src.exists():
        print(f"[SKIP] Not processed yet: {participant_id}/{block_id}")
        return 0

    copied = 0

    # Copy JSON files
    for filename in EXPORT_FILES:
        f = src / filename
        if f.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / filename)
            copied += 1

    # Copy feature CSVs (no videos)
    features_src = src / "features"
    if features_src.exists():
        for csv in sorted(features_src.rglob("*.csv")):
            rel = csv.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv, out)
            copied += 1

    print(f"[EXPORT] {participant_id}/{block_id} → {copied} files")
    return copied


def export_participant(processed_root: Path, export_root: Path, participant_id: str):
    p_dir = processed_root / participant_id
    if not p_dir.exists():
        print(f"[SKIP] No processed data for {participant_id}")
        return
    for block_dir in sorted(p_dir.iterdir()):
        if block_dir.is_dir():
            export_block(processed_root, export_root, participant_id, block_dir.name)


if __name__ == "__main__":
    cfg = load_config()
    processed_root = cfg["output_root"]
    export_root = processed_root.parent / "processed_novid"

    print(f"[EXPORT] Source:      {processed_root}")
    print(f"[EXPORT] Destination: {export_root}")

    target_participant = sys.argv[1] if len(sys.argv) > 1 else None
    target_block       = sys.argv[2] if len(sys.argv) > 2 else None

    if target_participant and target_block:
        export_block(processed_root, export_root, target_participant, target_block)
    elif target_participant:
        export_participant(processed_root, export_root, target_participant)
    else:
        # Export all participants
        if not processed_root.exists():
            print(f"[ERROR] processed root not found: {processed_root}")
            sys.exit(1)
        for p_dir in sorted(processed_root.iterdir()):
            if p_dir.is_dir():
                export_participant(processed_root, export_root, p_dir.name)

    print(f"\n[DONE] Export at: {export_root}")
