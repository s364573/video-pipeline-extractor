import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 20 AU Mapping with Plain English Explanations
AU_INFO = {
    "AU01": ("Inner Brow", "Surprise/Sadness"), "AU02": ("Outer Brow", "Surprise"),
    "AU04": ("Brow Lowerer", "Anger/Focus"), "AU05": ("Eye Widener", "Scare/Joy"),
    "AU06": ("Cheek Raiser", "Genuine Joy"), "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler", "AU10": "Upper Lip Raiser", "AU11": "Nasolabial",
    "AU12": ("Smile", "Happiness"), "AU14": "Dimpler", "AU15": "Lip Depressor",
    "AU17": "Chin Raiser", "AU20": "Lip Stretcher", "AU23": "Lip Tightener",
    "AU24": "Lip Pressor", "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Closed"
}
AU_COLS = list(AU_INFO.keys())

def get_stats(pid, block, quality):
    path = ROOT / "participant" / "processed" / pid / f"B{block}" / "features" / quality / "baseline_watching_statistics.csv"
    if not path.exists(): return None, None
    df = pd.read_csv(path).loc[:, ~pd.read_csv(path).columns.duplicated()]
    row = df[df["video"].str.contains("combined")].iloc[0]
    means = pd.to_numeric(row[[f"{c}_mean" for c in AU_COLS]], errors='coerce').fillna(0).values
    stds = pd.to_numeric(row[[f"{c}_std" for c in AU_COLS]], errors='coerce').fillna(0).values
    return means, stds

def get_top_joy_video(directory, mean, std):
    files = list(directory.glob("*.csv"))
    files = [f for f in files if "statistics" not in f.name]
    results = []
    idx_06, idx_12 = AU_COLS.index("AU06"), AU_COLS.index("AU12")
    
    for f in files:
        df = pd.read_csv(f)
        subset = df[[c for c in AU_COLS if c in df.columns]].fillna(0).values
        z = (subset - mean) / np.where(std == 0, 0.01, std)
        joy_signal = z[:, idx_06] + z[:, idx_12]
        # Smooth with rolling 10-frame window
        activation_series = pd.Series(joy_signal).rolling(10).mean()
        peak_score = activation_series.max()
        peak_idx = activation_series.idxmax()
        if pd.isna(peak_idx): peak_idx = 0
        results.append({"file": f, "score": peak_score, "z_data": z, 
                        "profile": z[int(peak_idx), :], "activation_line": activation_series})
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[0] if results else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="AGGG"); parser.add_argument("--p1_block", default="01")
    parser.add_argument("--p2", default="RRRR"); parser.add_argument("--p2_block", default="03")
    parser.add_argument("--quality", default="960_8fps")
    args = parser.parse_args()

    out_dir = ROOT / "analysis_results" / f"Diagnostic_{args.p1}_{args.p2}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    m1, s1 = get_stats(args.p1, "00", args.quality)
    m2, s2 = get_stats(args.p2, "00", args.quality)
    joy_p1 = get_top_joy_video(ROOT / "participant" / "processed" / args.p1 / f"B{args.p1_block}" / "features" / args.quality, m1, s1)
    joy_p2 = get_top_joy_video(ROOT / "participant" / "processed" / args.p2 / f"B{args.p2_block}" / "features" / args.quality, m2, s2)

    # --- RADAR DASHBOARD ---
    labels = [AU_INFO[au][0] if isinstance(AU_INFO[au], tuple) else AU_INFO[au] for au in AU_COLS]
    angles = np.concatenate((np.linspace(0, 2*np.pi, len(AU_COLS), endpoint=False), [0]))
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': 'polar'})
    
    # Left: Baseline explanation
    for m, s, label, color in [(m1, s1, args.p1, 'blue'), (m2, s2, args.p2, 'red')]:
        m_c = np.concatenate((m, [m[0]])); s_c = np.concatenate((s, [s[0]]))
        ax1.plot(angles, m_c, color=color, label=f"{label} Mean")
        ax1.fill_between(angles, np.maximum(0, m_c - s_c), m_c + s_c, color=color, alpha=0.1)
    ax1.set_title("BASELINE FINGERPRINT\nExplanations: Shows 'Resting' muscle tension.\nShaded area = natural fidgeting/noise.", pad=30)
    ax1.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)

    # Right: Peak Joy explanation
    for data, label, color in [(joy_p1, args.p1, 'blue'), (joy_p2, args.p2, 'red')]:
        prof = np.concatenate((data["profile"], [data["profile"][0]]))
        ax2.plot(angles, prof, color=color, label=f"{label} Peak Profile")
    ax2.set_title("PEAK JOY PROFILE\nExplanations: Shows deviations during the best Joy clip.\nStretches toward 'Smile' and 'Cheek Raiser'.", pad=30)
    ax2.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.savefig(out_dir / "radar_explanations.png", bbox_inches='tight')

    # --- HEATMAP DASHBOARD ---
    fig2, axes = plt.subplots(2, 2, figsize=(24, 14), gridspec_kw={'width_ratios': [1, 3]})
    
    # Plotting Logic for Heatmap and Activation
    for i, (data, p_id) in enumerate([(joy_p1, args.p1), (joy_p2, args.p2)]):
        # Activation Timeline (The "Activation Part")
        axes[i, 0].plot(data["activation_line"], color='green')
        axes[i, 0].fill_between(range(len(data["activation_line"])), data["activation_line"], color='green', alpha=0.2)
        axes[i, 0].set_title(f"{p_id} Activation Level Over Time")
        axes[i, 0].set_ylabel("Joy Intensity (Z-score)")
        
        # Heatmap
        im = axes[i, 1].imshow(data["z_data"].T, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
        axes[i, 1].set_title(f"{p_id} Muscle Movements | Clip: {data['file'].name}")
        axes[i, 1].set_yticks(range(len(AU_COLS)))
        axes[i, 1].set_yticklabels(labels)
        
        # EXPLANATION BOX
        box_text = f"PARTICIPANT: {p_id}\nCLIP: {data['file'].stem}\n\nHOW TO READ:\nRed = Active Muscle\nBlue = Suppressed/Relaxed\nRows = Specific Facial Muscles"
        axes[i, 1].text(1.02, 0.5, box_text, transform=axes[i, 1].transAxes, verticalalignment='center', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_dir / "joy_activation_dashboard.png")
    print(f"🏁 Dashboard Created in: {out_dir}")

if __name__ == "__main__":
    main()