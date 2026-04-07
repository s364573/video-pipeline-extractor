"""
run_all.py
----------
Batch-process multiple participants and blocks through the full pipeline.

Usage:
    python run_all.py                    # runs all participants/blocks defined below
    python run_all.py AAAD               # runs all blocks for one participant
    python run_all.py AAAD B01           # runs one specific block

Edit PARTICIPANTS and BLOCKS below to match your data.
Turn stages on/off with the RUN dict.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from main import process_block

# -------------------------
# CONFIGURE THESE
# -------------------------
PARTICIPANTS = [
    "AAAD",
    # "AABE",
    # "AACD",
    # add more here
]

BLOCKS = ["B00", "B01", "B02", "B03","B04","B05","B06"]

RUN = {
    "sync":       False,
    "clip":       False,
    "transcribe": False,
    "features":   True,
}

# -------------------------
# DISCOVER AVAILABLE BLOCKS
# -------------------------
def get_available_blocks(raw_root: Path, participant_id: str):
    """Returns block IDs that actually exist on disk for a participant."""
    participant_dir = raw_root / participant_id
    if not participant_dir.exists():
        return []
    found = []
    for block_id in BLOCKS:
        block_dir = participant_dir / block_id
        if block_dir.exists():
            found.append(block_id)
        # also check flat layout (files directly in participant dir)
        elif any((participant_dir / f"{participant_id}_{block_id}{ext}").exists()
                 for ext in [".mp4", ".csv", ".wav"]):
            found.append(block_id)
    return found


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    cfg = load_config()
    raw_root = cfg["raw_root"]

    # CLI overrides
    target_participant = sys.argv[1] if len(sys.argv) > 1 else None
    target_block       = sys.argv[2] if len(sys.argv) > 2 else None

    participants = [target_participant] if target_participant else PARTICIPANTS

    total = 0
    failed = []

    for participant_id in participants:
        blocks = [target_block] if target_block else get_available_blocks(raw_root, participant_id)

        if not blocks:
            print(f"[SKIP] No blocks found for {participant_id}")
            continue

        for block_id in blocks:
            print(f"\n{'='*50}")
            print(f"  Processing: {participant_id} / {block_id}")
            print(f"{'='*50}")
            try:
                process_block(participant_id, block_id, RUN=RUN)
                total += 1
            except Exception as e:
                print(f"[FAILED] {participant_id}/{block_id}: {e}")
                failed.append(f"{participant_id}/{block_id}")

    print(f"\n{'='*50}")
    print(f"Done. {total} blocks processed.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
