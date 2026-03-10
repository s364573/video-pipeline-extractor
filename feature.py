from pathlib import Path

import pandas as pd
import torch
from feat import Detector

device = "cpu" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")
device = "cpu"
# 1. Initialize the Detector (High accuracy settings)
detector = Detector(
    face_model="faceboxes",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    device=device,
)


def extract_features_standalone(pid, block_id):
    # Path to the already clipped videos
    block_dir = Path("participant") / pid / f"B{block_id}"

    if not block_dir.exists():
        print(f"Error: Folder {block_dir} does not exist. Run the clipper first.")
        return

    # Create the features sub-folder inside the block folder
    feat_dir = block_dir / "features"
    feat_dir.mkdir(exist_ok=True)

    # Find all mp4 files in the block folder
    video_files = sorted(list(block_dir.glob("*.mp4")))

    print(f"--- Starting Py-Feat Extraction for {pid} {block_id} ---")
    print(f"Found {len(video_files)} videos.")
    one = 0
    for video_path in video_files:
        one += 1
        output_csv = feat_dir / f"{video_path.stem}_features.csv"

        # Check if we already processed this video to save time
        if output_csv.exists():
            print(f"  > Skipping {video_path.name} (Already processed)")
            continue

        print(f"  > Analyzing: {video_path.name}")

        try:
            # Run Py-Feat
            # skip_frames=1 processes every frame.
            # Use skip_frames=2 to go 2x faster if you have a lot of data.
            predictions = detector.detect_video(
                str(video_path), skip_frames=2, batch_size=1, data_type="tensor"
            )

            # Save the result
            predictions.to_csv(output_csv)
            if one == 1:
                break
        except Exception as e:
            print(f"  ! Error processing {video_path.name}: {e}")

    print(f"\nFinished! All features are in: {feat_dir}")


# --- Run it for your specific block ---
extract_features_standalone("RRRR", "00")
