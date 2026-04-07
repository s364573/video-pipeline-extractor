import pandas as pd
import cv2
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# This looks for raw videos in your participant folder
RAW_VIDEO_ROOT = ROOT / "participant" 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p1", default="AGGG")
    parser.add_argument("--p2", default="RRRR")
    args = parser.parse_args()

    # Path to the NEW Clean Map
    map_dir = ROOT / "analysis_results" / f"Clean_Clustering_{args.p1}_{args.p2}" / f"K_{args.k}"
    map_path = map_dir / f"Clean_Map_K{args.k}.csv"
    
    gallery_dir = map_dir / "Cluster_Gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    if not map_path.exists():
        print(f"❌ Error: Could not find {map_path}. Run the clean cluster script first!")
        return

    df = pd.read_csv(map_path)
    print(f"📸 Extracting visual evidence for K={args.k}...\n")

    for pid in [args.p1, args.p2]:
        for cluster_id in range(args.k):
            # Find frames for this cluster
            subset = df[(df['participant'] == pid) & (df['cluster'] == cluster_id)]
            
            if subset.empty:
                continue
                
            # Sample up to 5 examples to get a better look at the "state"
            n_samples = min(5, len(subset))
            samples = subset.sample(n_samples)
            
            cluster_images = []
            
            for i, (_, row) in enumerate(samples.iterrows()):
                video_name = row['video_name']
                # Search for video extensions
                video_files = []
                for ext in ['.mp4', '.mov', '.avi', '.MP4']:
                    video_files.extend(list(RAW_VIDEO_ROOT.rglob(f"*{video_name}{ext}")))
                
                if not video_files:
                    continue
                
                cap = cv2.VideoCapture(str(video_files[0]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
                success, frame = cap.read()
                
                if success:
                    # Add labels to the image for clarity
                    label = f"{pid} | Cluster {cluster_id} | {video_name}"
                    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    img_path = gallery_dir / f"{pid}_C{cluster_id}_Ex{i}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    cluster_images.append(cv2.resize(frame, (480, 270))) # Store smaller version for grid
                
                cap.release()

            # Create a horizontal grid for this cluster/participant
            if cluster_images:
                grid = np.hstack(cluster_images)
                cv2.imwrite(str(gallery_dir / f"SUMMARY_{pid}_Cluster{cluster_id}.jpg"), grid)
                print(f"✅ Created summary grid for {pid} Cluster {cluster_id}")

    print(f"\n🏁 Finished! Check the images here:\n{gallery_dir}")

if __name__ == "__main__":
    main()