from pathlib import Path
import pandas as pd

features_dir = Path("participant/RRRR/B00/features")

# ---------------------------------
# TOGGLE BASELINE TYPE HERE
# ---------------------------------

BASELINE_TYPE = "speaking"   # "watching" or "speaking"

# ---------------------------------

if BASELINE_TYPE == "watching":
    csv_files = sorted(
        f for f in features_dir.glob("*.csv")
        if "VID" in f.stem
    )

elif BASELINE_TYPE == "speaking":
    csv_files = sorted(
        f for f in features_dir.glob("*.csv")
        if ("Q01" in f.stem or "Q02" in f.stem or "Q03" in f.stem)
    )

else:
    raise ValueError("BASELINE_TYPE must be 'watching' or 'speaking'")


# -----------------------
# columns to use
# -----------------------

frame = ["frame"]

au_cols = [
"AU01","AU02","AU04","AU05","AU06","AU07",
"AU09","AU10","AU11","AU12","AU14","AU15",
"AU17","AU20","AU23","AU24","AU25","AU26",
"AU28","AU43"
]

emotion_cols = [
"anger","disgust","fear","happiness",
"sadness","surprise","neutral"
]

cols = frame + au_cols + emotion_cols


rows = []
all_frames = []

for file in csv_files:

    df = pd.read_csv(file)

    df = df[cols]

    row = {"video": file.stem + f"_{BASELINE_TYPE}_baseline"}

    for c in cols:
        row[f"{c}_mean"] = df[c].mean()
        row[f"{c}_std"] = df[c].std()

    rows.append(row)
    all_frames.append(df)


# -----------------------
# combined baseline
# -----------------------

combined = pd.concat(all_frames, ignore_index=True)

combined_row = {"video": f"RRRR_B00_combined_{BASELINE_TYPE}_baseline"}

for c in cols:
    combined_row[f"{c}_mean"] = combined[c].mean()
    combined_row[f"{c}_std"] = combined[c].std()

rows.append(combined_row)


dataset = pd.DataFrame(rows)

output_file = features_dir / f"baseline_{BASELINE_TYPE}_statistics.csv"

dataset.to_csv(output_file, index=False)

print("Saved:", output_file)
print(dataset.head())