"""
explore_aus.py
--------------
Explores AU values from py-feat CSVs.
Produces:
  1. Summary table — mean, std, max, % active per AU
  2. Feature selection recommendation
  3. Per-clip AU timeline plot
  4. Emotion prediction overview

Run with venv_feat from src/:
    python explore_aus.py

Or change PARTICIPANT_ID / BLOCK_ID at the bottom.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config_loader import load_config

# -------------------------
# AU REFERENCE TABLE
# -------------------------
AU_REFERENCE = {
    "AU01": ("Inner Brow Raise",     "Sadness, Fear",          1),
    "AU02": ("Outer Brow Raise",     "Surprise, Fear",         1),
    "AU04": ("Brow Lowerer",         "Anger, Sadness",         1),
    "AU05": ("Upper Lid Raiser",     "Fear, Surprise",         2),
    "AU06": ("Cheek Raiser",         "Genuine joy",            1),
    "AU07": ("Lid Tightener",        "Anger, Disgust",         2),
    "AU09": ("Nose Wrinkler",        "Disgust",                2),
    "AU10": ("Upper Lip Raiser",     "Disgust",                2),
    "AU11": ("Nasolabial Deepener",  "Disgust",                3),
    "AU12": ("Lip Corner Puller",    "Smile/Happiness",        1),
    "AU14": ("Dimpler",              "Subtle smile",           3),
    "AU15": ("Lip Corner Depr.",     "Sadness",                1),
    "AU17": ("Chin Raiser",          "Sadness, Disgust",       2),
    "AU20": ("Lip Stretcher",        "Fear",                   2),
    "AU23": ("Lip Tightener",        "Anger",                  2),
    "AU24": ("Lip Pressor",          "Anger, Suppression",     2),
    "AU25": ("Lips Part",            "Speech/Many emotions",   2),
    "AU26": ("Jaw Drop",             "Surprise, Fear",         2),
    "AU28": ("Lip Suck",             "Suppression",            3),
    "AU43": ("Eyes Closed",          "Disgust, Sadness",       3),
}

EMOTION_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


# -------------------------
# LOAD
# -------------------------
def load_block_features(base_path: Path, clip_type="stimuli"):
    feature_dirs = sorted((base_path / "features").iterdir())
    
    # pick the latest config (skip5 over skip2)
    config_dir = feature_dirs[-1]
    clip_dir   = config_dir / clip_type

    print(f"[INFO] Reading from: {clip_dir}")
    csvs = sorted(clip_dir.glob("*.csv"))
    print(f"[INFO] Loading {len(csvs)} CSVs")

    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["clip"] = csv.stem
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def get_au_cols(df):
    return [c for c in df.columns if c.startswith("AU")]


# -------------------------
# SUMMARY TABLE
# -------------------------
def print_summary(df):
    au_cols = get_au_cols(df)

    print("\n" + "="*95)
    print(f"{'AU':<7} {'Muscle':<28} {'Mean':>6} {'Std':>6} {'Max':>6} {'%>1.5':>7}  {'Tier'}  Signal")
    print("="*95)

    rows = []
    for au in au_cols:
        vals       = df[au].dropna()
        mean       = vals.mean()
        std        = vals.std()
        max_       = vals.max()
        pct_active = (vals > 0.3).mean() * 100
        ref        = AU_REFERENCE.get(au, ("Unknown", "?", 3))
        rows.append(dict(au=au, muscle=ref[0], signal=ref[1],
                         mean=mean, std=std, max=max_,
                         pct=pct_active, tier=ref[2]))

    rows.sort(key=lambda r: r["pct"], reverse=True)

    for r in rows:
        stars = "T1" if r["tier"] == 1 else ("T2" if r["tier"] == 2 else "T3")
        print(f"{r['au']:<7} {r['muscle']:<28} "
              f"{r['mean']:>6.2f} {r['std']:>6.2f} {r['max']:>6.2f} "
              f"{r['pct']:>6.1f}%  {stars:<4}  {r['signal']}")

    print("="*95)
    return rows


# -------------------------
# FEATURE SELECTION
# -------------------------
def recommend(rows):
    print("\n--- FEATURE SELECTION ---")
    keep, drop = [], []
    for r in rows:
        ref  = AU_REFERENCE.get(r["au"], ("", "", 3))
        tier = ref[2]
        if tier == 1 and r["pct"] > 1.0:
            keep.append(r["au"])
        elif tier == 2 and r["pct"] > 5.0:
            keep.append(r["au"])
        else:
            drop.append(r["au"])

    print(f"\n  KEEP ({len(keep)}): {keep}")
    print(f"  DROP ({len(drop)}): {drop}")
    print("\n  Note: Based on B00 neutral baseline.")
    print("  Re-evaluate after running emotion blocks.")
    return keep, drop


# -------------------------
# PLOT: AU TIMELINE PER CLIP
# -------------------------
def plot_clip_aus(df, clip_name, keep_aus, save_dir: Path):
    if not keep_aus:
        keep_aus = get_au_cols(df)

    clip_df = df[df["clip"] == clip_name].copy()
    if clip_df.empty:
        print(f"[WARN] Clip not found: {clip_name}")
        return

    aus_present = [a for a in keep_aus if a in clip_df.columns]
    time        = clip_df["time"].values
    n           = len(aus_present)

    fig, axes = plt.subplots(n, 1, figsize=(14, n * 1.4), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"AU Timeline  —  {clip_name}", fontsize=12, fontweight="bold")

    for ax, au in zip(axes, aus_present):
        ref   = AU_REFERENCE.get(au, ("?", "?", 3))
        color = "#e74c3c" if ref[2] == 1 else ("#3498db" if ref[2] == 2 else "#95a5a6")

        ax.fill_between(time, clip_df[au].values, alpha=0.25, color=color)
        ax.plot(time, clip_df[au].values, color=color, linewidth=0.9)
        ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticks([0, 2.5, 5])
        ax.tick_params(labelsize=7)
        ax.text(1.002, 0.5, ref[0], transform=ax.transAxes,
                fontsize=7, va="center", color=color)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{clip_name}_aus.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out.name}")


# -------------------------
# PLOT: EMOTION OVERVIEW
# -------------------------
def plot_emotions_overview(df, clips, save_dir: Path):
    emo_cols = [c for c in EMOTION_COLS if c in df.columns]
    if not emo_cols:
        print("[WARN] No emotion columns found")
        return

    n_clips = len(clips)
    fig, axes = plt.subplots(n_clips, 1, figsize=(14, n_clips * 2.2), sharex=False)
    if n_clips == 1:
        axes = [axes]

    fig.suptitle("Emotion Predictions Over Time (py-feat)", fontsize=12, fontweight="bold")
    colors = ["#e74c3c","#8e44ad","#3498db","#2ecc71","#1abc9c","#f39c12","#95a5a6"]

    for ax, clip in zip(axes, clips):
        clip_df = df[df["clip"] == clip]
        time    = clip_df["time"].values
        for emo, color in zip(emo_cols, colors):
            ax.plot(time, clip_df[emo].values, label=emo, color=color, linewidth=1.0)
        ax.set_title(clip, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Prob", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=7, ncol=len(emo_cols))
    plt.tight_layout()

    out = save_dir / "emotions_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out.name}")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PARTICIPANT_ID = "AAAD"
    BLOCK_ID       = "B00"
    CLIP_TYPE      = "stimuli"

    cfg       = load_config()
    base_path = cfg["output_root"] / PARTICIPANT_ID / BLOCK_ID
    save_dir  = base_path / "plots"

    print(f"[INFO] base_path: {base_path}")

    df      = load_block_features(base_path, CLIP_TYPE)
    au_cols = get_au_cols(df)
    clips   = sorted(df["clip"].unique())

    print(f"[INFO] {len(au_cols)} AUs | {len(clips)} clips | {len(df)} total frames")

    # 1. Summary table
    rows = print_summary(df)

    # 2. Feature selection
    keep, drop = recommend(rows)

    # 3. AU timelines per clip
    print("\n[PLOTS] AU timelines...")
    for clip in clips:
        plot_clip_aus(df, clip, keep, save_dir / "au_timelines")

    # 4. Emotion overview
    print("\n[PLOTS] Emotion overview...")
    plot_emotions_overview(df, clips, save_dir)

    print(f"\n[DONE] Plots saved to: {save_dir}")
