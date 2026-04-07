import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

# FOCUS ONLY ON "EXPRESSIVE" AUs TO REDUCE NOISE
EYE_AUS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU43"]
MOUTH_AUS = ["AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26"]
ALL_CORE_AUS = EYE_AUS + MOUTH_AUS

AU_LOOKUP = {
    "AU01": "Inner Brow", "AU02": "Outer Brow", "AU04": "Brow Low",
    "AU05": "Eye Wide", "AU06": "Cheek Raiser", "AU07": "Lid Tight",
    "AU09": "Nose Wrink", "AU10": "Up Lip", "AU11": "Nasolabial",
    "AU12": "Smile", "AU14": "Dimple", "AU15": "Lip Depress",
    "AU17": "Chin Raise", "AU20": "Lip Stretch", "AU23": "Lip Tight",
    "AU24": "Lip Press", "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Close"
}

def get_stats(pid, quality):
    path = ROOT / "participant" / "processed" / pid / "B00" / "features" / quality / "baseline_watching_statistics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline stats at {path}")
    df = pd.read_csv(path)
    row = df[df["video"].str.contains("combined")].iloc[0]
    m = pd.to_numeric(row[[f"{c}_mean" for c in ALL_CORE_AUS]], errors='coerce').fillna(0).values
    s = pd.to_numeric(row[[f"{c}_std" for c in ALL_CORE_AUS]], errors='coerce').fillna(0).values
    return m, s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="AGGG")
    parser.add_argument("--p1_block", default="01")
    parser.add_argument("--p2", default="RRRR")
    parser.add_argument("--p2_block", default="03")
    parser.add_argument("--quality", default="960_8fps")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    out_dir = ROOT / "analysis_results" / f"Clean_Clustering_{args.p1}_{args.p2}" / f"K_{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD AND FILTER DATA
    m1, s1 = get_stats(args.p1, args.quality)
    m2, s2 = get_stats(args.p2, args.quality)
    p_data = []
    
    for pid, block, m, s in [(args.p1, args.p1_block, m1, s1), (args.p2, args.p2_block, m2, s2)]:
        path = ROOT / "participant" / "processed" / pid / f"B{block}" / "features" / args.quality
        for f in sorted(path.glob("*VID.csv")):
            df = pd.read_csv(f)[ALL_CORE_AUS].fillna(0).values
            z = (df - m) / np.where(s == 0, 0.01, s)
            
            # --- NOISE THRESHOLD ---
            # Ignore anything less than 1.2 standard deviations (Neutral territory)
            z[np.abs(z) < 1.2] = 0
            
            # Smooth with a rolling window to stabilize clusters
            z_smoothed = pd.DataFrame(z).rolling(window=5, center=True).mean().fillna(0).values
            
            p_data.append({'pid': pid, 'vid_name': f.stem, 'z_data': z_smoothed})

    all_z = np.vstack([d['z_data'] for d in p_data])

    # 2. CLUSTERING
    km = KMeans(n_clusters=args.k, random_state=42, n_init=15)
    labels = km.fit_predict(all_z)

    # 3. SAVE THE CLEAN MAP
    map_data = []
    curr = 0
    for d in p_data:
        num_frames = len(d['z_data'])
        vid_labels = labels[curr : curr + num_frames]
        for i in range(num_frames):
            map_data.append({
                'participant': d['pid'], 'video_name': d['vid_name'],
                'frame_index': i, 'cluster': int(vid_labels[i])
            })
        curr += num_frames
    
    pd.DataFrame(map_data).to_csv(out_dir / f"Clean_Map_K{args.k}.csv", index=False)

    # 4. EXPLANATION FIGURE (Bar Chart of Active AUs)
    fig, axes = plt.subplots(1, args.k, figsize=(4*args.k, 5))
    if args.k == 1: axes = [axes]
    
    for i in range(args.k):
        center = km.cluster_centers_[i]
        # Get labels for the plot
        clean_labels = [AU_LOOKUP.get(au, au) for au in ALL_CORE_AUS]
        
        axes[i].barh(clean_labels, center, color='teal')
        axes[i].set_title(f"Cluster {i}\nProfile")
        axes[i].set_xlim(-5, 5) # Show deviation from baseline
        axes[i].tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "Cluster_Definitions.png")
    print(f"🏁 Finished! Results and Clean Map are in: {out_dir}")

if __name__ == "__main__":
    main()