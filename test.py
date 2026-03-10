import csv
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

import read_csv as rc

# Map the detail strings to your numeric codes
EMO_MAP = {
    "NEUTRAL": "00",
    "NEU": "00",
    "SURPRISE": "01",
    "JOY": "02",
    "DISGUST": "03",
    "FEAR": "04",
    "SADNESS": "05",
    "ANGER": "06",
}


def get_offset(wav_path, csv_beep_time):
    """Calculates the time difference between CSV log and Video Audio."""
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)

    # Bandpass 1kHz
    nyq = 0.5 * sr
    b, a = butter(4, [900 / nyq, 1100 / nyq], btype="band")
    filtered = filtfilt(b, a, audio)

    # Energy envelope
    window = int(0.05 * sr)
    energy = np.convolve(filtered**2, np.ones(window) / window, mode="same")

    # Detect start
    threshold = 0.3 * energy.max()
    start_index = np.where(energy > threshold)[0][0]
    video_beep_time = start_index / sr

    return video_beep_time - float(csv_beep_time)


def clip_video(input_file, start, duration, output_path):
    """Executes FFmpeg command."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_file),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd)


def process_block(pid, block_id):
    # 1. Load Files
    files = rc.load_participant_files(pid, block_id)
    if not all(files.values()):
        print(f"Missing files for {pid} {block_id}")
        return

    # 2. Setup Output Directory
    out_dir = Path("participant") / pid / f"B{block_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Sync Offset
    csv_beeps = rc.extract_beep_times(files["csv"])
    offset = get_offset(files["wav"], csv_beeps[0])
    print(f"--- Processing {pid} Block {block_id} (Offset: {offset:.4f}s) ---")

    # 4. Parse CSV for Stimuli
    with open(files["csv"], newline="") as f:
        rows = list(csv.DictReader(f))

    stim_index = 0
    for i, row in enumerate(rows):
        if row["event"] == "video_start":
            stim_index += 1
            detail = row["detail"]
            detail_up = detail.upper()

            # Identify emotion code from the EMO_MAP
            emo_code = "99"
            for key, code in EMO_MAP.items():
                if key in detail_up:
                    emo_code = code
                    break

            try:
                # Find video end row to get duration
                end_row = next(
                    r
                    for r in rows[i:]
                    if r["event"] == "video_end" and r["detail"] == row["detail"]
                )

                v_start = float(row["t"]) + offset
                v_dur = (float(end_row["t"]) - float(row["t"])) + 5.0

                # UPDATED: Sequence (Sxx) now comes before Emotion Code
                v_name = f"{pid}_B{block_id}_S{stim_index:02d}_{emo_code}_VID.mp4"
                print(f"  > [{stim_index:02d}] Clipping: {detail} -> {v_name}")
                clip_video(files["mp4"], v_start, v_dur, out_dir / v_name)

                # 2. Clip Questions (Dynamic Loop)
                q_count = 0
                for j in range(i + 1, len(rows)):
                    # Stop looking if we hit the next video
                    if rows[j]["event"] == "video_start":
                        break

                    if rows[j]["event"] == "question_start":
                        q_count += 1
                        q_start = float(rows[j]["t"]) + offset

                        # Find question duration
                        q_end_row = rows[j + 1]
                        q_dur = float(q_end_row["t"]) - float(rows[j]["t"])

                        # UPDATED: Sequence (Sxx) now comes before Emotion Code
                        q_name = f"{pid}_B{block_id}_S{stim_index:02d}_{emo_code}_Q{q_count:02d}.mp4"
                        clip_video(files["mp4"], q_start, q_dur, out_dir / q_name)

            except (StopIteration, IndexError):
                continue

    print(f"Done! Files saved to {out_dir}\n")


# Example Run
process_block("RRRR", "03")
