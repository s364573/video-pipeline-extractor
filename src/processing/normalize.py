
"""
normalize.py
------------
Z-score normalizes AU features against each participant's B00 neutral baseline.

Output per participant/block:
    Data/processed/AAAD/B01/features/<config>/stimuli/<clip>_normalized.csv

Usage:
    python processing/normalize.py

Each AU value becomes:
    z = (value - participant_neutral_mean) / participant_neutral_std

Z > 2.0 = significant deviation from neutral.
Z < -1.0 = suppression below neutral.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
import pandas as pd
import numpy as np
from utils.config_loader import load_config

# -------------------------
# CONFIG
# -------------------------
AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU11", "AU12", "AU14", "AU15",
    "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43"
]

POSE_COLS   = ["Pitch", "Roll", "Yaw"]
NEUTRAL_BLOCK = "B00"


# -------------------------
# LOAD FEATURES
# -------------------------
def get_feature_dir(base_path: Path) -> Path:
    """Get the latest feature config dir."""
    feat_dirs = sorted((base_path / "features").iterdir())
    if not feat_dirs:
        raise FileNotFoundError(f"No features found at {base_path}/features/")
    return feat_dirs[-1]  # latest config


def load_clip_csvs(feature_dir: Path, clip_type: str) -> pd.DataFrame:
    clip_dir = feature_dir / clip_type
    if not clip_dir.exists():
        return pd.DataFrame()

    csvs = sorted(clip_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()

    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["clip"] = csv.stem
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# -------------------------
# COMPUTE NEUTRAL BASELINE
# -------------------------
def compute_neutral_baseline(participant_path: Path) -> dict:
    """
    Compute per-AU mean and std from the B00 neutral block.
    Returns dict: {AU01: {"mean": x, "std": y}, ...}
    """
    b00_path    = participant_path / NEUTRAL_BLOCK
    feature_dir = get_feature_dir(b00_path)

    df = load_clip_csvs(feature_dir, "stimuli")
    if df.empty:
        raise ValueError(f"No B00 stimuli features found at {b00_path}")

    baseline = {}
    for au in AU_COLS:
        if au not in df.columns:
            continue
        vals = df[au].dropna()
        baseline[au] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()) if vals.std() > 1e-6 else 1e-6
        }

    # also baseline pose
    for col in POSE_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        baseline[col] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()) if vals.std() > 1e-6 else 1e-6
        }

    print(f"[NORM] Baseline computed from {len(df)} frames ({NEUTRAL_BLOCK})")
    return baseline


# -------------------------
# APPLY Z-SCORE
# -------------------------
def zscore_df(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Apply Z-score normalization to AU and pose columns."""
    out = df.copy()

    for col, stats in baseline.items():
        if col not in out.columns:
            continue
        out[f"{col}_z"] = (out[col] - stats["mean"]) / stats["std"]

    return out


# -------------------------
# DEVIATION SCORE
# -------------------------
# Weighted sum of Z-scores across Tier 1 AUs
# Gives one scalar per frame: how far from neutral is this face?

TIER1_AUS = ["AU01", "AU02", "AU04", "AU06", "AU12", "AU15"]
TIER1_WEIGHTS = {
    "AU01": 1.0,  # inner brow — sadness/fear signal
    "AU02": 1.0,  # outer brow — surprise/fear
    "AU04": 1.5,  # brow lowerer — anger key
    "AU06": 2.0,  # cheek raiser — Duchenne smile, very discriminative
    "AU12": 2.0,  # lip corner — smile
    "AU15": 1.5,  # lip corner depressor — sadness key
}

def compute_deviation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted deviation score per frame.
    Uses absolute Z-scores so both activation and suppression count.
    """
    out   = df.copy()
    score = pd.Series(0.0, index=df.index)

    for au, weight in TIER1_WEIGHTS.items():
        z_col = f"{au}_z"
        if z_col in df.columns:
            score += df[z_col].abs() * weight

    out["deviation_score"] = score
    return out


# -------------------------
# PROCESS ONE BLOCK
# -------------------------
def normalize_block(participant_path: Path, block_id: str, baseline: dict):
    block_path  = participant_path / block_id
    feature_dir = get_feature_dir(block_path)

    for clip_type in ["stimuli", "questions"]:
        clip_dir = feature_dir / clip_type
        if not clip_dir.exists():
            continue

        out_dir = feature_dir / f"{clip_type}_normalized"
        out_dir.mkdir(parents=True, exist_ok=True)

        csvs = sorted(clip_dir.glob("*.csv"))
        print(f"[NORM] {block_id}/{clip_type}: {len(csvs)} clips")

        for csv in csvs:
            df = pd.read_csv(csv)

            df = zscore_df(df, baseline)
            df = compute_deviation_score(df)

            out_path = out_dir / csv.name
            df.to_csv(out_path, index=False)

        print(f"[NORM] Saved to: {out_dir}")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PARTICIPANT_ID = "AAAD"
    BLOCKS         = ["B01", "B02", "B03", "B04", "B05", "B06"]

    cfg              = load_config()
    participant_path = cfg["output_root"] / PARTICIPANT_ID

    print(f"[NORM] Participant: {PARTICIPANT_ID}")

    # 1. Compute neutral baseline from B00
    baseline = compute_neutral_baseline(participant_path)

    print(f"[NORM] Baseline AUs: {list(baseline.keys())}")

    # 2. Also normalize B00 itself (useful for validation)
    normalize_block(participant_path, NEUTRAL_BLOCK, baseline)

    # 3. Normalize all emotion blocks
    for block_id in BLOCKS:
        block_path = participant_path / block_id
        if not block_path.exists():
            print(f"[SKIP] {block_id} not found")
            continue
        normalize_block(participant_path, block_id, baseline)

    print("[NORM] Done.")
