import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AU_LOOKUP = {
    "AU01": "Inner Brow Raiser", "AU02": "Outer Brow Raiser", "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser", "AU06": "Cheek Raiser", "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler", "AU10": "Upper Lip Raiser", "AU11": "Nasolabial Deepener",
    "AU12": "Smile (Lip Puller)", "AU14": "Dimpler", "AU15": "Lip Depressor",
    "AU17": "Chin Raiser", "AU20": "Lip Stretcher", "AU23": "Lip Tightener",
    "AU24": "Lip Pressor", "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Closed"
}
AU_COLS = list(AU_LOOKUP.keys())

def get_stats(pid, block, quality):
    path = ROOT / "participant" / "processed" / pid / f"B{block}" / "features" / quality / "baseline_watching_statistics.csv"
    if not path.exists():
        print(f"❌ ERROR: Baseline file not found for {pid} at: {path}")
        return None, None
    df = pd.read_csv(path)
    row = df[df["video"].str.contains("combined")].iloc[0]
    means = pd.to_numeric(row[[f"{c}_mean" for c in AU_COLS]], errors='coerce').fillna(0)
    stds = pd.to_numeric(row[[f"{c}_std" for c in AU_COLS]], errors='coerce').fillna(0.01)
    return means, stds

def get_top_videos(directory, mean, std, count=2):
    # Try multiple common patterns to find your files
    files = list(directory.glob("*VID*.csv")) + list(directory.glob("*vid*.csv"))
    files = list(set(files)) # Remove duplicates
    
    print(f"🔎 Searching in: {directory}")
    print(f"✅ Found {len(files)} VID files.")
    
    if not files:
        return []
        
    results = []
    for f in files:
        df = pd.read_csv(f)
        subset = df[[c for c in AU_COLS if c in df.columns]].fillna(0)
        z = (subset - mean.values) / std.values
        activation = np.sqrt((z**2).sum(axis=1)).rolling(10).mean().max()
        results.append((f, activation, z))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:count]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="AGGG")
    parser.add_argument("--p1_block", default="01")
    parser.add_argument("--p2", default="RRRR")
    parser.add_argument("--p2_block", default="03")
    parser.add_argument("--quality", default="960_8fps")
    args = parser.parse_args()

    out_dir = ROOT / "analysis_results" / f"Diagnostic_{args.p1}_vs_{args.p2}"
    out_dir.mkdir(parents=True, exist_ok=True)

    m1, s1 = get_stats(args.p1, "00", args.quality)
    m2, s2 = get_stats(args.p2, "00", args.quality)
    if m1 is None or m2 is None: return

    # --- RADAR CHART ---
    labels = [AU_LOOKUP[au] for au in AU_COLS]
    angles = np.linspace(0, 2*np.pi, len(AU_COLS), endpoint=False).tolist()
    angles += angles[:1]
    
    fig_radar = plt.figure(figsize=(10, 10))
    ax_r = fig_radar.add_subplot(111, polar=True)
    for m, label, color in [(m1, args.p1, 'blue'), (m2, args.p2, 'red')]:
        vals = m.tolist() + m.tolist()[:1]
        ax_r.plot(angles, vals, color=color, linewidth=2, label=f"{label} Neutral")
        ax_r.fill(angles, vals, color=color, alpha=0.1)
    ax_r.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    plt.title("Baseline Facial Profiles (Fingerprint)", y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig(out_dir / "baseline_radar.png", bbox_inches='tight')

    # --- TOP VIDEOS ---
    p1_dir = ROOT / "participant" / "processed" / args.p1 / f"B{args.p1_block}" / "features" / args.quality
    p2_dir = ROOT / "participant" / "processed" / args.p2 / f"B{args.p2_block}" / "features" / args.quality
    
    top_p1 = get_top_videos(p1_dir, m1, s1)
    top_p2 = get_top_videos(p2_dir, m2, s2)

    rows = max(len(top_p1), len(top_p2))
    if rows == 0:
        print("❌ CRITICAL: No videos found to plot. Check your folder paths in the console above.")
        return

    fig, axes = plt.subplots(rows, 2, figsize=(20, 6*rows), squeeze=False)
    for i in range(rows):
        for col, data_list, p_id in [(0, top_p1, args.p1), (1, top_p2, args.p2)]:
            if i < len(data_list):
                f, score, z = data_list[i]
                im = axes[i, col].imshow(z.T, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
                axes[i, col].set_title(f"{p_id} Peak {i+1}: {f.stem}")
                axes[i, col].set_yticks(range(len(AU_COLS)))
                axes[i, col].set_yticklabels(AU_COLS)
            else:
                axes[i, col].axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / "top_peaks.png")
    print(f"🏁 DONE. Check results in: {out_dir}")

if __name__ == "__main__":
    main()