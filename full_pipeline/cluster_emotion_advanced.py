import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

AU_LOOKUP = {
    "AU01": "Inner Brow", "AU02": "Outer Brow", "AU04": "Brow Low",
    "AU05": "Eye Wide", "AU06": "Cheek Raiser", "AU07": "Lid Tight",
    "AU09": "Nose Wrink", "AU10": "Up Lip", "AU11": "Nasolabial",
    "AU12": "Smile", "AU14": "Dimple", "AU15": "Lip Depress",
    "AU17": "Chin Raise", "AU20": "Lip Stretch", "AU23": "Lip Tight",
    "AU24": "Lip Press", "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Close"
}
AU_COLS = list(AU_LOOKUP.keys())

def get_stats(pid, quality):
    path = ROOT / "participant" / "processed" / pid / "B00" / "features" / quality / "baseline_watching_statistics.csv"
    df = pd.read_csv(path)
    row = df[df["video"].str.contains("combined")].iloc[0]
    m = pd.to_numeric(row[[f"{c}_mean" for c in AU_COLS]], errors='coerce').fillna(0).values
    s = pd.to_numeric(row[[f"{c}_std" for c in AU_COLS]], errors='coerce').fillna(0).values
    return m, s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="AGGG"); parser.add_argument("--p1_block", default="01")
    parser.add_argument("--p2", default="RRRR"); parser.add_argument("--p2_block", default="03")
    parser.add_argument("--quality", default="960_8fps")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    # Organized Output Folder
    out_dir = ROOT / "analysis_results" / f"Clustering_Study_{args.p1}_{args.p2}" / f"K_{args.k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    m1, s1 = get_stats(args.p1, args.quality)
    m2, s2 = get_stats(args.p2, args.quality)
    p_data = []
    for pid, block, m, s in [(args.p1, args.p1_block, m1, s1), (args.p2, args.p2_block, m2, s2)]:
        path = ROOT / "participant" / "processed" / pid / f"B{block}" / "features" / args.quality
        for f in sorted(path.glob("*VID.csv")):
            df = pd.read_csv(f)[AU_COLS].fillna(0).values
            z = (df - m) / np.where(s == 0, 0.01, s)
            p_data.append({'pid': pid, 'vid_name': f.stem.replace('_VID', '').replace('PID_', ''), 'z_data': z})

    all_z = np.vstack([d['z_data'] for d in p_data])

    

    # 2. ANALYSIS: CLUSTERING & PCA
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = km.fit_predict(all_z)  # This creates 'labels'
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_z)

    # 2. SAVE THE CLUSTER MAP
    map_data = []
    curr = 0
    for d in p_data:
        num_frames = len(d['z_data'])
        for i in range(num_frames):
            map_data.append({
                'participant': d['pid'],
                'video_name': d['vid_name'],
                'frame_index': i,
                'cluster': labels[curr + i]  # FIXED: changed 'all_labels' to 'labels'
            })
        curr += num_frames
    
    map_df = pd.DataFrame(map_data)
    map_df.to_csv(out_dir / f"Cluster_Map_K{args.k}.csv", index=False)
    print(f"📂 Cluster Map saved: {out_dir}/Cluster_Map_K{args.k}.csv")

    # Distribute labels/coords back
    curr = 0
    for d in p_data:
        d['labels'] = labels[curr:curr + len(d['z_data'])]
        d['coords'] = coords[curr:curr + len(d['z_data'])]
        curr += len(d['z_data'])

    # --- FIGURE 1: 2D PCA MAP (The "Galaxy" View) ---
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10', args.k)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=colors, s=2, alpha=0.4)
    plt.colorbar(scatter, ticks=range(args.k), label='Cluster ID')
    plt.title(f"2D Facial State Map (K={args.k}) | Each dot is 1 frame\nGroups represent distinct behavioral modes")
    plt.xlabel("Principal Component 1 (Major Movement)"); plt.ylabel("Principal Component 2 (Nuance)")
    plt.savefig(out_dir / f"PCA_Map_K{args.k}.png", bbox_inches='tight')

    # --- FIGURE 2: PREVALENCE TIMELINES (The "Barcode" View) ---
    fig, axes = plt.subplots(len(p_data), 1, figsize=(20, 1.2 * len(p_data)), sharex=True)
    if len(p_data) == 1: axes = [axes]
    for i, d in enumerate(p_data):
        axes[i].imshow(d['labels'].reshape(1, -1), aspect='auto', cmap=colors, vmin=0, vmax=args.k-1)
        axes[i].set_ylabel(f"{d['pid']}\n{d['vid_name']}", rotation=0, labelpad=50, fontweight='bold', ha='center', va='center')
        axes[i].set_yticks([])
    
    legend_patches = [mpatches.Patch(color=colors(i), label=f'Cluster {i}') for i in range(args.k)]
    fig.legend(handles=legend_patches, loc='upper right', title="Facial States")
    plt.suptitle(f"Temporal Prevalence Barcodes (K={args.k})", fontsize=16, y=0.95)
    plt.savefig(out_dir / f"Timeline_Barcodes_K{args.k}.png", bbox_inches='tight')

    # --- FIGURE 3: CLUSTER ANATOMY RADARS (The "Details" View) ---
    angles = np.concatenate((np.linspace(0, 2*np.pi, len(AU_COLS), endpoint=False), [0]))
    fig_r, axes_r = plt.subplots(1, args.k, figsize=(5*args.k, 6), subplot_kw={'projection': 'polar'})
    if args.k == 1: axes_r = [axes_r]
    for i in range(args.k):
        ax = axes_r[i]
        for pid, color in [(args.p1, 'blue'), (args.p2, 'red')]:
            p_frames = np.vstack([d['z_data'][d['labels'] == i] for d in p_data if d['pid'] == pid and (d['labels'] == i).any()] or [np.zeros(len(AU_COLS))])
            if p_frames.any():
                prof = np.concatenate((p_frames.mean(axis=0), [p_frames.mean(axis=0)[0]]))
                ax.plot(angles, prof, color=color, label=f"P: {pid}", linewidth=2); ax.fill(angles, prof, color=color, alpha=0.1)
        ax.set_thetagrids(np.degrees(angles[:-1]), list(AU_LOOKUP.values()), fontsize=7)
        ax.set_title(f"CLUSTER {i} PROFILE", fontweight='bold', pad=15)
    plt.legend(loc='lower center', bbox_to_anchor=(-args.k/2 + 0.5, -0.25), ncol=2)
    plt.savefig(out_dir / f"Radar_Profiles_K{args.k}.png", bbox_inches='tight')

    print(f"🏁 Finished K={args.k}. Results: {out_dir}")

if __name__ == "__main__":
    main()