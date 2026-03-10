from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# ROOT PATH (makes script work anywhere)
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

participant = "RRRR"

baseline_block = "00"
video_block = "03"

BASELINE_TYPE = "watching"   # "watching" or "speaking"


# ------------------------------------------------
# PATHS
# ------------------------------------------------

baseline_dir = ROOT / "participant" / "processed" /participant  / f"B{baseline_block}" / "features"

video_dir = ROOT / "participant" / "processed" / participant / f"B{video_block}" / "features"

print("Baseline dir:", baseline_dir)
print("Video dir:", video_dir)


# ------------------------------------------------
# SIGNAL DEFINITIONS
# ------------------------------------------------

signals_au = [
"AU01","AU02","AU04","AU05","AU06","AU07",
"AU09","AU10","AU11","AU12","AU14","AU15",
"AU17","AU20","AU23","AU24","AU25","AU26",
"AU28","AU43"
]

emotion_cols = [
"anger","disgust","fear","happiness",
"sadness","surprise","neutral"
]


# ------------------------------------------------
# LOAD BASELINE
# ------------------------------------------------

baseline_file = baseline_dir / f"baseline_{BASELINE_TYPE}_statistics.csv"

if not baseline_file.exists():
    raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

baseline = pd.read_csv(baseline_file)

baseline_name = f"{participant}_B{baseline_block}_combined_{BASELINE_TYPE}_baseline"

baseline_row = baseline[baseline["video"] == baseline_name]

if baseline_row.empty:
    raise ValueError("Baseline row not found inside baseline file")

mean = baseline_row[[f"{c}_mean" for c in signals_au]].iloc[0]
std  = baseline_row[[f"{c}_std"  for c in signals_au]].iloc[0]

mean.index = signals_au
std.index = signals_au

std = std.replace(0, np.nan)


# ------------------------------------------------
# FIND VIDEO FILES
# ------------------------------------------------

video_files = sorted(video_dir.glob("*VID.csv"))

print("Found videos:", len(video_files))

scores = []


# ------------------------------------------------
# EVALUATE EACH VIDEO
# ------------------------------------------------

for file in video_files:

    df = pd.read_csv(file)

    if "frame" not in df.columns:
        continue

    df[signals_au] = df[signals_au].interpolate(method="linear")
    df[signals_au] = df[signals_au].bfill()
    df[signals_au] = df[signals_au].ffill()

    z = (df[signals_au] - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna(axis=1, how="all")

    if z.empty:
        continue

    activation = np.sqrt((z**2).sum(axis=1))

    score = activation.max()

    scores.append((file, score))


if len(scores) == 0:
    raise RuntimeError("No valid videos found")


# ------------------------------------------------
# SELECT TOP 2
# ------------------------------------------------

scores.sort(key=lambda x: x[1], reverse=True)

top_videos = scores[:2]

print("\nTop emotional activations:")
for file, score in top_videos:
    print(file.name, "score:", round(score,3))


# ------------------------------------------------
# PLOT
# ------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14,8))

for i, (file, score) in enumerate(top_videos):

    df = pd.read_csv(file)

    df[signals_au] = df[signals_au].interpolate(method="linear")
    df[signals_au] = df[signals_au].bfill()
    df[signals_au] = df[signals_au].ffill()

    z = (df[signals_au] - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna(axis=1, how="all")

    activation = np.sqrt((z**2).sum(axis=1))
    activation = pd.Series(activation).rolling(5, center=True).mean()

    # -----------------------------
    # activation timeline
    # -----------------------------

    ax1 = axes[i,0]

    ax1.plot(df["frame"], activation)

    ax1.axhline(2, linestyle="--")
    ax1.axhline(3, linestyle="--")

    ax1.set_title(f"{file.stem}\nActivation score: {score:.2f}")
    ax1.set_ylabel("Activation")


    # -----------------------------
    # emotion probabilities
    # -----------------------------

    ax2 = axes[i,1]

    for emo in emotion_cols:
        if emo in df.columns:
            ax2.plot(df["frame"], df[emo], label=emo)

    ax2.set_title("Emotion prediction")
    ax2.set_xlabel("Frame")

    if i == 0:
        ax2.legend()


plt.tight_layout()
plt.show()

# -----------------------------
# LARGE AU HEATMAP
# -----------------------------

fig2, axes2 = plt.subplots(
    len(top_videos),
    1,
    figsize=(20, 12),     # much bigger
    dpi=120
)

if len(top_videos) == 1:
    axes2 = [axes2]

for i, (file, score) in enumerate(top_videos):

    df = pd.read_csv(file)

    df[signals_au] = df[signals_au].interpolate(method="linear")
    df[signals_au] = df[signals_au].bfill()
    df[signals_au] = df[signals_au].ffill()

    z = (df[signals_au] - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.fillna(0)

    heat = z[signals_au].T

    ax = axes2[i]

    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-3,
        vmax=3
    )

    ax.set_title(
        f"{file.stem}  |  Activation Score: {score:.2f}",
        fontsize=14,
        pad=10
    )

    ax.set_ylabel("Facial Action Units", fontsize=12)

    ax.set_yticks(range(len(signals_au)))
    ax.set_yticklabels(signals_au, fontsize=11)

    ax.set_xlabel("Frame", fontsize=12)

    ax.tick_params(axis="x", labelsize=10)

# color bar
cbar = fig2.colorbar(im, ax=axes2, fraction=0.02, pad=0.02)
cbar.set_label("Deviation from Neutral Baseline (Z-score)", fontsize=12)

plt.tight_layout()
plt.show()