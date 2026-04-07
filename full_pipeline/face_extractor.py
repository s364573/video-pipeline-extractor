import pandas as pd
import cv2
import argparse
import numpy as np
from pathlib import Path

# --- SETUP PATHS ---
ROOT = Path(__file__).resolve().parent.parent
# This looks for the raw videos in your participant/processed directory 
# or wherever you keep the original files
RAW_VIDEO_ROOT = ROOT / "participant"

def extract_frames():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p1", default="AGGG")
    parser.add_argument("--p2", default="RRRR")
    args = parser.parse_args()

    # Path to the map we created in the clustering step
    map_dir = ROOT / "analysis_results" / f"Clustering_Study_{args.p1}_{args.p2}" / f"K_{args.k}"
    map_path = map_dir / f"Cluster_Map_K{args.k}.csv"
    
    # Create a folder for the photos
    gallery_dir = map_dir / "Cluster_Gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    if not map_path.exists():
        print(f"❌ Error: Run your master clustering script first!")
        return

    df = pd.read_csv(map_path)
    print(f"📸 Starting Auto-Extraction for K={args.k}...\n")

    for pid in [args.p1, args.p2]:
        print(f"--- Processing {pid} ---")
        for cluster_id in range(args.k):
            # Find frames for this specific cluster and participant
            matches = df[(df['participant'] == pid) & (df['cluster'] == cluster_id)]
            
            if not matches.empty:
                # Sample 3 random examples (or fewer if cluster is small)
                n_samples = min(3, len(matches))
                samples = matches.sample(n_samples)
                
                for i, (_, row) in enumerate(samples.iterrows()):
                    # Find the video file (searching for .mp4, .mov, .avi)
                    video_name = row['video_name']
                    video_files = []
                    for ext in ['*.mp4', '*.mov', '*.avi', '*.MP4']:
                        video_files.extend(list(RAW_VIDEO_ROOT.rglob(f"*{video_name}{ext}")))
                    
                    if not video_files:
                        print(f"  ⚠️ Video not found: {video_name}")
                        continue
                    
                    video_path = video_files[0]
                    cap = cv2.VideoCapture(str(video_path))
                    
                    # Jump to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
                    success, frame = cap.read()
                    
                    if success:
                        img_name = f"{pid}_C{cluster_id}_Ex{i}_{video_name}.jpg"
                        # Resize for easy viewing in slides (optional)
                        # frame = cv2.resize(frame, (640, 360)) 
                        cv2.imwrite(str(gallery_dir / img_name), frame)
                        print(f"  ✅ Saved: Cluster {cluster_id} | {img_name}")
                    
                    cap.release()

    print(f"\n🏁 Finished! Your 'Reality Check' photos are in:\n{gallery_dir}")

if __name__ == "__main__":
    extract_frames()