"""
segment.py
----------
Finds facial expression events (clusters of deviation from neutral)
in normalized AU data and builds LoRA training pairs.

Output:
    Data/processed/AAAD/training_pairs.json

Each training pair contains:
    - video clip path
    - start/end timestamps of the expression event
    - dominant emotion
    - temporal description of AU dynamics
    - prompt for LoRA training

Usage:
    python processing/segment.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # adds src/ to path
import pandas as pd
import numpy as np
import json
from utils.config_loader import load_config

# -------------------------
# CONFIG
# -------------------------
DEVIATION_THRESHOLD = 2.0   # Z-score units — significant deviation
MIN_EVENT_FRAMES    = 5     # minimum frames to count as an expression event
MERGE_GAP_FRAMES    = 8     # merge events closer than this many frames
SMOOTH_WINDOW       = 5     # rolling average window for deviation score

AU_REFERENCE = {
    "AU01": ("Inner Brow Raise",  "sadness/fear"),
    "AU02": ("Outer Brow Raise",  "surprise/fear"),
    "AU04": ("Brow Lowerer",      "anger/sadness"),
    "AU06": ("Cheek Raiser",      "genuine joy"),
    "AU07": ("Lid Tightener",     "anger/disgust"),
    "AU09": ("Nose Wrinkler",     "disgust"),
    "AU12": ("Lip Corner Puller", "happiness"),
    "AU15": ("Lip Corner Depr.", "sadness"),
    "AU17": ("Chin Raiser",       "sadness/disgust"),
    "AU20": ("Lip Stretcher",     "fear"),
    "AU23": ("Lip Tightener",     "anger"),
    "AU24": ("Lip Pressor",       "anger/suppression"),
    "AU25": ("Lips Part",         "general"),
    "AU26": ("Jaw Drop",          "surprise/fear"),
}

EMOTION_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


# -------------------------
# LOAD NORMALIZED CSVs
# -------------------------
def load_normalized(base_path: Path, block_id: str, clip_type="stimuli") -> list[dict]:
    """Load all normalized CSVs for a block. Returns list of {clip, df}."""
    feature_dirs = sorted((base_path / block_id / "features").iterdir())
    if not feature_dirs:
        return []

    config_dir = feature_dirs[-1]
    norm_dir   = config_dir / f"{clip_type}_normalized"

    if not norm_dir.exists():
        print(f"[WARN] No normalized data: {norm_dir}")
        return []

    results = []
    for csv in sorted(norm_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        results.append({"clip": csv.stem, "df": df, "path": csv})

    return results


# -------------------------
# SMOOTH DEVIATION
# -------------------------
def smooth_deviation(df: pd.DataFrame) -> pd.Series:
    if "deviation_score" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["deviation_score"].rolling(SMOOTH_WINDOW, center=True).mean().fillna(0)


# -------------------------
# FIND EXPRESSION EVENTS
# -------------------------
def find_events(df: pd.DataFrame) -> list[dict]:
    """
    Find continuous windows where deviation score exceeds threshold.
    Returns list of {start_frame, end_frame, start_time, end_time, peak_frame}.
    """
    smoothed = smooth_deviation(df)
    active   = smoothed > DEVIATION_THRESHOLD

    events   = []
    in_event = False
    start    = None

    for i, val in enumerate(active):
        if val and not in_event:
            in_event = True
            start    = i
        elif not val and in_event:
            in_event = False
            end      = i - 1
            if (end - start) >= MIN_EVENT_FRAMES:
                events.append({"start_frame": start, "end_frame": end})

    # close any open event
    if in_event:
        end = len(active) - 1
        if (end - start) >= MIN_EVENT_FRAMES:
            events.append({"start_frame": start, "end_frame": end})

    # merge close events
    merged = []
    for ev in events:
        if merged and (ev["start_frame"] - merged[-1]["end_frame"]) <= MERGE_GAP_FRAMES:
            merged[-1]["end_frame"] = ev["end_frame"]
        else:
            merged.append(ev)

    # add time and peak
    for ev in merged:
        segment  = df.iloc[ev["start_frame"]:ev["end_frame"] + 1]
        smoothed_seg = smoothed.iloc[ev["start_frame"]:ev["end_frame"] + 1]

        ev["start_time"] = float(df.iloc[ev["start_frame"]]["time"])
        ev["end_time"]   = float(df.iloc[ev["end_frame"]]["time"])
        ev["duration"]   = round(ev["end_time"] - ev["start_time"], 3)
        ev["peak_frame"] = int(smoothed_seg.idxmax())
        ev["peak_time"]  = float(df.loc[ev["peak_frame"], "time"])
        ev["peak_score"] = float(smoothed_seg.max())

    return merged


# -------------------------
# DESCRIBE AU DYNAMICS
# -------------------------
def describe_event(df: pd.DataFrame, event: dict) -> dict:
    """
    Describe the temporal dynamics of an expression event.
    Returns dict with dominant AU, onset speed, peak AUs, emotion prediction.
    """
    seg = df.iloc[event["start_frame"]:event["end_frame"] + 1].copy()

    # Z-score columns present
    z_cols = [c for c in seg.columns if c.endswith("_z") and c.replace("_z", "") in AU_REFERENCE]

    # Peak AU activations
    peak_aus = {}
    for z_col in z_cols:
        au   = z_col.replace("_z", "")
        peak = seg[z_col].max()
        if peak > 2.0:
            peak_aus[au] = round(float(peak), 2)

    peak_aus = dict(sorted(peak_aus.items(), key=lambda x: x[1], reverse=True))

    # Dominant emotion from py-feat predictions
    dominant_emotion = "neutral"
    if all(c in seg.columns for c in EMOTION_COLS):
        emo_means   = seg[EMOTION_COLS].mean()
        dom_idx     = emo_means.idxmax()
        if emo_means[dom_idx] > 0.2:
            dominant_emotion = dom_idx

    # Onset speed: frames from start to peak
    frames_to_peak  = event["peak_frame"] - event["start_frame"]
    fps_effective   = 10  # 50fps / skip5
    onset_sec       = round(frames_to_peak / fps_effective, 2)

    onset_type = "sudden" if onset_sec < 0.3 else ("gradual" if onset_sec < 1.0 else "slow")

    # Head pose at peak
    pose = {}
    for col in ["Pitch", "Roll", "Yaw"]:
        z_col = f"{col}_z"
        if z_col in seg.columns:
            val = float(seg.loc[event["peak_frame"], z_col]) if event["peak_frame"] in seg.index else 0.0
            if abs(val) > 1.0:
                pose[col] = round(val, 2)

    return {
        "dominant_emotion": dominant_emotion,
        "peak_aus":         peak_aus,
        "onset_type":       onset_type,
        "onset_sec":        onset_sec,
        "pose_deviation":   pose,
    }


# -------------------------
# BUILD PROMPT
# -------------------------
def build_prompt(clip: str, event: dict, description: dict) -> str:
    """
    Build a temporal text prompt for LoRA training.
    Format: "Norwegian adult, [emotion], [onset], peak at [time]s,
             AU[X] ([muscle]) peaks at [z], valence=[v], intensity=[i]"
    """
    emo    = description["dominant_emotion"]
    onset  = description["onset_type"]
    t_start = event["start_time"]
    t_peak  = event["peak_time"]
    t_end   = event["end_time"]

    # Top 3 AUs
    top_aus = list(description["peak_aus"].items())[:3]
    au_str  = ", ".join(
        f"{au} ({AU_REFERENCE.get(au, ('?','?'))[0]}, z={z})"
        for au, z in top_aus
    )

    # Pose
    pose_parts = [f"{k} shift z={v}" for k, v in description["pose_deviation"].items()]
    pose_str   = (", " + ", ".join(pose_parts)) if pose_parts else ""

    prompt = (
        f"Norwegian adult expressing {emo}, {onset} onset at {t_start:.1f}s, "
        f"peak expression at {t_peak:.1f}s, fades by {t_end:.1f}s. "
        f"Active AUs: {au_str}{pose_str}. "
        f"Clip: {clip}."
    )

    return prompt


# -------------------------
# PROCESS ONE BLOCK
# -------------------------
def process_block(participant_path: Path, block_id: str, clip_type="stimuli") -> list[dict]:
    clips   = load_normalized(participant_path, block_id, clip_type)
    pairs   = []

    print(f"\n[SEG] {block_id}/{clip_type}: {len(clips)} clips")

    for item in clips:
        clip = item["clip"]
        df   = item["df"]

        events = find_events(df)

        if not events:
            print(f"  [SKIP] {clip} — no expression events found")
            continue

        print(f"  [CLIP] {clip}: {len(events)} events")

        for i, event in enumerate(events):
            desc   = describe_event(df, event)
            prompt = build_prompt(clip, event, desc)

            pair = {
                "participant": str(participant_path.name),
                "block":       block_id,
                "clip":        clip,
                "event_id":    f"{clip}_ev{i:02d}",
                "clip_type":   clip_type,
                "start_time":  event["start_time"],
                "end_time":    event["end_time"],
                "duration":    event["duration"],
                "peak_time":   event["peak_time"],
                "peak_score":  event["peak_score"],
                "dominant_emotion": desc["dominant_emotion"],
                "peak_aus":    desc["peak_aus"],
                "onset_type":  desc["onset_type"],
                "pose":        desc["pose_deviation"],
                "prompt":      prompt,
            }

            pairs.append(pair)
            print(f"    ev{i:02d}: {desc['dominant_emotion']} | {event['start_time']:.1f}s→{event['end_time']:.1f}s | {prompt[:80]}...")

    return pairs


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PARTICIPANT_ID = "AAAD"
    BLOCKS         = ["B00", "B01", "B02", "B03", "B04", "B05", "B06"]

    cfg              = load_config()
    participant_path = cfg["output_root"] / PARTICIPANT_ID

    all_pairs = []

    for block_id in BLOCKS:
        block_path = participant_path / block_id
        if not block_path.exists():
            print(f"[SKIP] {block_id} not found")
            continue

        pairs = process_block(participant_path, block_id, "stimuli")
        all_pairs.extend(pairs)

    # Save
    out_path = participant_path / "training_pairs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(all_pairs)} training pairs saved to: {out_path}")

    # Summary
    from collections import Counter
    emotions = Counter(p["dominant_emotion"] for p in all_pairs)
    print("\n[SUMMARY] Emotion distribution:")
    for emo, count in emotions.most_common():
        print(f"  {emo:<12} {count}")
