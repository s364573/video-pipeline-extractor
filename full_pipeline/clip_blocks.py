import csv
import subprocess
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


# ------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "participant" / "raw"
PROCESSED_DIR = ROOT / "participant" / "processed"


# ------------------------------------------------
# EMOTION MAP
# ------------------------------------------------

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


# ------------------------------------------------
# LOAD RAW FILES
# ------------------------------------------------

def load_participant_files(pid, block):

    folder = RAW_DIR / pid

    files = {"mp4": None, "wav": None, "csv": None}

    for f in folder.glob("*"):

        if f.is_dir():
            continue

        name = f.stem.upper()

        if not name.startswith(f"{pid}_{block}"):
            continue

        ext = f.suffix.lower()

        if ext == ".mp4":
            files["mp4"] = f

        elif ext == ".wav":
            files["wav"] = f

        elif ext == ".csv":
            files["csv"] = f

    return files


# ------------------------------------------------
# EXTRACT BEEP TIMES
# ------------------------------------------------

def extract_beep_times(csv_file):

    times = []

    with open(csv_file, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["event"] in ["beep_start", "beep_end"]:
                times.append(float(row["t"]))

    return times


# ------------------------------------------------
# AUTOMATIC SYNC OFFSET
# ------------------------------------------------

def get_offset(wav_path, csv_beep_time):

    sr, audio = wavfile.read(wav_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(float)

    nyq = 0.5 * sr

    b, a = butter(4, [900/nyq, 1100/nyq], btype="band")

    filtered = filtfilt(b, a, audio)

    window = int(0.05 * sr)

    energy = np.convolve(filtered**2, np.ones(window)/window, mode="same")

    threshold = 0.3 * energy.max()

    start_index = np.where(energy > threshold)[0][0]

    video_beep_time = start_index / sr

    return video_beep_time - csv_beep_time


# ------------------------------------------------
# FFMPEG CLIPPING
# ------------------------------------------------

def clip_video(input_file, start, duration, output):

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(input_file),
        "-c", "copy",
        str(output)
    ]

    subprocess.run(cmd)


# ------------------------------------------------
# PROCESS BLOCK
# ------------------------------------------------

def process_block(pid, block, manual_offset=None):

    files = load_participant_files(pid, block)

    if not all(files.values()):
        print("Missing files:", files)
        return

    print("\nUsing files:")
    print(files)

    out_dir = PROCESSED_DIR / pid / f"B{block}"
    out_dir.mkdir(parents=True, exist_ok=True)

    beep_times = extract_beep_times(files["csv"])

    # -----------------------------
    # OFFSET SELECTION
    # -----------------------------

    if manual_offset is not None:

        offset = manual_offset
        print("Using MANUAL offset:", offset)

    else:

        offset = get_offset(files["wav"], beep_times[0])
        print("Detected offset:", round(offset,3))


    print(f"\nProcessing {pid} block {block}")
    print("Final Offset:", round(offset,3))

    with open(files["csv"], newline="") as f:
        rows = list(csv.DictReader(f))

    stim_index = 0

    for i, row in enumerate(rows):

        if row["event"] != "video_start":
            continue

        stim_index += 1

        detail = row["detail"].upper()

        emo_code = "99"

        for key, code in EMO_MAP.items():

            if key in detail:
                emo_code = code
                break

        try:

            end_row = next(
                r for r in rows[i:]
                if r["event"] == "video_end"
                and r["detail"] == row["detail"]
            )

            v_start = float(row["t"]) + offset
            v_dur = (float(end_row["t"]) - float(row["t"])) + 5

            v_name = f"{pid}_B{block}_S{stim_index:02d}_{emo_code}_VID.mp4"

            print("Clipping", v_name)

            clip_video(files["mp4"], v_start, v_dur, out_dir / v_name)

            # --------------------------------
            # QUESTIONS
            # --------------------------------

            q_count = 0

            for j in range(i + 1, len(rows)):

                if rows[j]["event"] == "video_start":
                    break

                if rows[j]["event"] == "question_start":

                    q_count += 1

                    q_start = float(rows[j]["t"]) + offset

                    q_end = next(
                        r for r in rows[j:]
                        if r["event"] == "question_end"
                    )

                    q_dur = float(q_end["t"]) - float(rows[j]["t"])

                    q_name = f"{pid}_B{block}_S{stim_index:02d}_{emo_code}_Q{q_count:02d}.mp4"

                    clip_video(files["mp4"], q_start, q_dur, out_dir / q_name)

        except StopIteration:
            continue

    print("\nSaved clips to:", out_dir)


# ------------------------------------------------
# TERMINAL ENTRY
# ------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("participant")
    parser.add_argument("block")

    parser.add_argument(
        "--manual_offset",
        type=float,
        default=None,
        help="manually override sync offset in seconds"
    )

    args = parser.parse_args()

    process_block(
        args.participant,
        args.block,
        args.manual_offset
    )