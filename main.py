import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

import read_csv as rc

path = rc.load_participant_files("RRRR", "00")

# ---- load file ----
sr, audio = wavfile.read("RRRR_0-ATTT0.wav")
beep_times = rc.extract_beep_times(path["csv"])

# mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio = audio.astype(float)

# ---- bandpass 1kHz ----
nyq = 0.5 * sr
b, a = butter(4, [900 / nyq, 1100 / nyq], btype="band")  # type: ignore
filtered = filtfilt(b, a, audio)

# ---- energy envelope ----
window = int(0.05 * sr)
energy = np.convolve(filtered**2, np.ones(window) / window, mode="same")

# ---- detect start ----
threshold = 0.3 * energy.max()
start_index = np.where(energy > threshold)[0][0]
video_beep = start_index / sr
print(start_index / sr)
offset = video_beep - float(beep_times[0])

start = offset + float(beep_times[0])
cmd = [
    "ffmpeg",
    "-i",
    path["mp4"],
    "-ss",
    str(start),
    "-t",
    "10",
    "-c",
    "copy",
    "RRRR_00_beep.mp4",
]
subprocess.run(cmd)
