from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features_dir = Path("participant/RRRR/B00/features")
video_dir = Path("participant/RRRR/B03/features")

# ---------------------------------
BASELINE_TYPE = "watching"
# ---------------------------------

baseline_file = features_dir / f"baseline_{BASELINE_TYPE}_statistics.csv"
baseline = pd.read_csv(baseline_file)

video = pd.read_csv(video_dir / "RRRR_B03_S08_01_VID.csv")


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

signals = signals_au


# -----------------------------
# Fix missing values
# -----------------------------
video[signals] = video[signals].interpolate(method="linear")
video[signals] = video[signals].bfill()
video[signals] = video[signals].ffill()


# -----------------------------
# Load baseline
# -----------------------------
baseline_name = f"RRRR_B00_combined_{BASELINE_TYPE}_baseline"

baseline_row = baseline[baseline["video"] == baseline_name]

mean = baseline_row[[f"{c}_mean" for c in signals]].iloc[0]
std  = baseline_row[[f"{c}_std"  for c in signals]].iloc[0]

mean.index = signals
std.index = signals

std = std.replace(0, np.nan)


# -----------------------------
# Z-score AU deviation
# -----------------------------
z = (video[signals] - mean) / std

z = z.replace([np.inf, -np.inf], np.nan)
z = z.dropna(axis=1, how="all")


# -----------------------------
# Find strongest AU
# -----------------------------
largest_signal = z.abs().max().sort_values(ascending=False).index[0]

print("Using baseline:", BASELINE_TYPE)
print("Strongest signal:", largest_signal)


# -----------------------------
# Smooth AU signal
# -----------------------------
z[largest_signal] = z[largest_signal].rolling(5, center=True).mean()


# -----------------------------
# Plot
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)

# ---- AU deviation ----

ax1.plot(video["frame"], z[largest_signal], label=largest_signal)

ax1.axhline(2, linestyle="--")
ax1.axhline(-2, linestyle="--")

ax1.set_ylabel("AU Z-score")
ax1.set_title(f"{largest_signal} deviation from {BASELINE_TYPE} baseline")
ax1.legend()


# ---- Emotion probabilities ----

for emo in emotion_cols:
    ax2.plot(video["frame"], video[emo], label=emo)

ax2.set_ylabel("Emotion probability")
ax2.set_xlabel("Frame")
ax2.legend(loc="upper right")


plt.tight_layout()
plt.show()