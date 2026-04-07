import subprocess
import tempfile
from pathlib import Path
import json
from faster_whisper import WhisperModel
from utils.config_loader import load_config, get_device
import os

_cfg = load_config()
FFMPEG = _cfg["ffmpeg"]
DEVICE = get_device()

os.environ["PATH"] = str(Path(FFMPEG).parent) + os.pathsep + os.environ["PATH"]

# -----------------------------
# CONFIG
# -----------------------------
# faster-whisper only supports "cpu" and "cuda" (no MPS backend)
# int8 on CPU is fast and accurate enough
MODEL_SIZE    = "large-v2"
LANGUAGE      = "no"
FW_DEVICE     = "cuda" if DEVICE == "cuda" else "cpu"
COMPUTE_TYPE  = "float16" if FW_DEVICE == "cuda" else "int8"

NUM_MAP = {
    "en": 1, "ett": 1,
    "to": 2,
    "tre": 3,
    "fire": 4,
    "fem": 5
}


# -----------------------------
# AUDIO PREPROCESSING
# -----------------------------
def preprocess_audio(input_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)

    cmd = [
        FFMPEG,
        "-y",
        "-i", str(input_path),
        "-af",
        "silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB,"
        "silenceremove=stop_periods=1:stop_silence=0.3:stop_threshold=-40dB,"
        "loudnorm",
        str(tmp_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_path


# -----------------------------
# WHISPER TRANSCRIPTION
# -----------------------------
def transcribe_file(model: WhisperModel, audio_path: Path):
    processed = preprocess_audio(audio_path)

    segments_gen, info = model.transcribe(
        str(processed),
        language=LANGUAGE,
        task="transcribe",

        # fast settings — responses are max 6 words
        temperature=0.0,
        beam_size=3,
        best_of=1,

        condition_on_previous_text=False,
        max_initial_timestamp=1.0,

        # hallucination control
        compression_ratio_threshold=2.0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.3,

        initial_prompt="Kort svar på norsk. Ett ord eller kort setning.",
    )

    # faster-whisper returns a generator — must consume it
    segments = list(segments_gen)

    good_segments = [
        s for s in segments
        if s.avg_logprob > -2.0
        and s.no_speech_prob < 0.6
        and s.compression_ratio < 2.0
    ]

    if not good_segments:
        good_segments = segments

    text = " ".join(s.text.strip() for s in good_segments).strip()
    final_text = good_segments[-1].text.strip() if good_segments else ""
    final_text = final_text.split(".")[0].strip()

    return {
        "text": text,
        "final_text": final_text,
        "segments": [
            {
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
            }
            for s in good_segments
        ],
        "language": info.language,
    }


# -----------------------------
# POST-PROCESSING
# -----------------------------
def normalize_number(text: str):
    t = text.lower().strip()
    if t in NUM_MAP:
        return NUM_MAP[t]
    for k in NUM_MAP:
        if k in t:
            return NUM_MAP[k]
    return None


def is_valid(text: str):
    if not text:
        return False
    if len(text.split()) > 6:
        return False
    if len(text) > 4 and text.count(text[:2]) > 3:
        return False
    return True


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def transcribe_labels(labels, base_path: Path, model: WhisperModel):
    for clip in labels:
        if clip["type"] != "response":
            continue

        clip_path = base_path / clip["file"]
        if not clip_path.exists():
            clip_path = clip_path.with_suffix(".MP4")
        if not clip_path.exists():
            print(f"[WARN] Missing: {clip_path}")
            continue

        print(f"[ASR] {clip_path.name}")

        try:
            result = transcribe_file(model, clip_path)
            final_text = result["final_text"]
            clip["transcription"] = {
                "text": result["text"],
                "final_text": final_text,
                "number": normalize_number(final_text),
                "valid": is_valid(final_text),
                "language": result["language"]
            }
        except Exception as e:
            print(f"[ERROR] {clip_path}: {e}")
            clip["transcription"] = {"error": str(e)}

    return labels


# -----------------------------
# ENTRY
# -----------------------------
def load_model() -> WhisperModel:
    print(f"[ASR] Loading faster-whisper {MODEL_SIZE} ({FW_DEVICE}, {COMPUTE_TYPE})...")
    return WhisperModel(MODEL_SIZE, device=FW_DEVICE, compute_type=COMPUTE_TYPE)


def main():
    model = load_model()
    base_path = Path("PATH_TO_CLIPS")

    with open("labels.json") as f:
        labels = json.load(f)

    labels = transcribe_labels(labels, base_path, model)

    with open("labels_with_transcriptions.json", "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
