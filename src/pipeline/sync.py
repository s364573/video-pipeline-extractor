import numpy as np
import subprocess
import tempfile
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import spectrogram
from utils.config_loader import load_config
_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]

DEFAULT_TONE_DURATION = 1.0
SEARCH_WINDOW_START = 0.0
SEARCH_WINDOW_DURATION = 45.0
DETECTION_SAMPLE_RATE = 8000
FRAME_SECONDS = 0.03
HOP_SECONDS = 0.005
LOW_FREQ = 980
HIGH_FREQ = 1020
ADJACENT_BAND_PADDING = 230
MIN_TONE_RATIO = 4.0


def extract_audio_window_to_wav(
    media_path: Path,
    start_time: float,
    duration: float,
    sample_rate: int = DETECTION_SAMPLE_RATE,
) -> Path:
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(media_path),
        "-ss", f"{max(0.0, start_time):.3f}",
        "-t", f"{max(0.1, duration):.3f}",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        "-acodec", "pcm_s16le",
        str(tmp_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to extract audio")

    return tmp_path


def detect_beep_time(
    audio_path: Path,
    clip_start: float = 0.0,
    beep_duration: float = DEFAULT_TONE_DURATION,
    low_freq: int = LOW_FREQ,
    high_freq: int = HIGH_FREQ,
    frame_seconds: float = FRAME_SECONDS,
    hop_seconds: float = HOP_SECONDS,
):
    sr, audio = wavfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak <= 1e-8:
        raise ValueError(f"Audio segment is silent: {audio_path}")
    audio /= peak

    nperseg = min(max(int(sr * frame_seconds), 64), len(audio))
    hop_samples = max(int(sr * hop_seconds), 1)
    noverlap = min(max(nperseg - hop_samples, 0), nperseg - 1)

    freqs, times, spectrum = spectrogram(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    adjacent_mask = (
        ((freqs >= low_freq - ADJACENT_BAND_PADDING) & (freqs < low_freq))
        | ((freqs > high_freq) & (freqs <= high_freq + ADJACENT_BAND_PADDING))
    )

    if not band_mask.any():
        raise ValueError("Target beep band is not represented in the spectrogram")
    if not adjacent_mask.any():
        raise ValueError("Adjacent comparison band is not represented in the spectrogram")

    band_energy = spectrum[band_mask].mean(axis=0) + 1e-12
    adjacent_energy = spectrum[adjacent_mask].mean(axis=0) + 1e-12
    tone_ratio = band_energy / adjacent_energy
    tone_score = band_energy * tone_ratio

    window_bins = max(1, int(round(beep_duration / hop_seconds)))
    window_bins = min(window_bins, len(tone_score))
    smoothed_score = np.convolve(
        tone_score,
        np.ones(window_bins, dtype=np.float32) / window_bins,
        mode="valid",
    )

    best_idx = int(np.argmax(smoothed_score))
    best_window = slice(best_idx, best_idx + window_bins)
    local_peak_idx = best_idx + int(np.argmax(tone_score[best_window]))
    dominant_freqs = freqs[band_mask]
    dominant_spectrum = spectrum[band_mask][:, best_window].mean(axis=1)
    dominant_freq = float(dominant_freqs[int(np.argmax(dominant_spectrum))])

    onset_time = clip_start + max(0.0, float(times[best_idx]) - frame_seconds / 2)
    end_idx = best_idx + window_bins - 1
    beep_end_time = clip_start + float(times[end_idx]) + frame_seconds / 2
    peak_time = clip_start + float(times[local_peak_idx])
    best_score = float(smoothed_score[best_idx])
    peak_ratio = float(tone_ratio[local_peak_idx])

    if peak_ratio < MIN_TONE_RATIO:
        raise ValueError(
            f"Could not find a reliable 1 kHz sync tone in {audio_path} (peak ratio={peak_ratio:.2f})"
        )

    print(
        f"[SYNC] freq={dominant_freq:.0f} Hz | start={onset_time:.3f}s | "
        f"peak={peak_time:.3f}s | ratio={peak_ratio:.2f}"
    )

    return {
        "beep_start_time": onset_time,
        "beep_peak_time": peak_time,
        "beep_end_time": beep_end_time,
        "dominant_freq": dominant_freq,
        "detection_score": best_score,
        "peak_ratio": peak_ratio,
    }


def get_media_source(block):
    if block.wav_path.exists():
        return block.wav_path

    if block.video_path.exists():
        return block.video_path

    raise FileNotFoundError(
        f"No audio or video source found for {block.participant_id} {block.block_id}"
    )


def get_event_time(events, event_name, required=True):
    matches = events.loc[events["event"] == event_name, "t"]
    if matches.empty:
        if required:
            raise ValueError(f"Missing required event: {event_name}")
        return None
    return float(matches.iloc[0])


def compute_sync_offset(block, events):
    csv_beep_time = get_event_time(events, "beep_start")
    csv_beep_end = get_event_time(events, "beep_end", required=False)
    csv_duration = (
        (csv_beep_end - csv_beep_time) if csv_beep_end is not None else DEFAULT_TONE_DURATION
    )
    beep_duration = max(DEFAULT_TONE_DURATION, min(csv_duration, 2.0))

    clip_start = SEARCH_WINDOW_START
    clip_duration = SEARCH_WINDOW_DURATION

    audio_source = get_media_source(block)
    audio_path = extract_audio_window_to_wav(audio_source, clip_start, clip_duration)

    try:
        detection = detect_beep_time(
            audio_path,
            clip_start=clip_start,
            beep_duration=beep_duration,
        )
    finally:
        audio_path.unlink(missing_ok=True)

    video_beep_time = detection["beep_start_time"]
    offset = video_beep_time - csv_beep_time

    return {
        "video_beep_time": video_beep_time,
        "video_beep_peak_time": detection["beep_peak_time"],
        "video_beep_end_time": detection["beep_end_time"],
        "csv_beep_time": csv_beep_time,
        "offset": offset,
        "dominant_freq": detection["dominant_freq"],
        "detection_score": detection["detection_score"],
        "peak_ratio": detection["peak_ratio"],
        "search_window_start": clip_start,
        "search_window_duration": clip_duration,
    }
