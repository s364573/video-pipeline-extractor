import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # adds src/ to path
from pathlib import Path
import json
import copy
import subprocess
import sys
from data.loader import Session
from data.parser import get_event_pairs
from data.trials import build_trials_with_questions, build_metadata
from pipeline.sync import compute_sync_offset
from pipeline.clip import clip_trials
from pipeline.transcribe import transcribe_labels
from utils.config_loader import load_config, get_device      # ← from utils/
import whisper

# -------------------------
# PATHS FROM CONFIG
# -------------------------
cfg = load_config()
RAW_ROOT = cfg["raw_root"]
OUTPUT   = cfg["output_root"]

FEAT_PYTHON          = cfg["project_root"].parent / "venv_feat" / "Scripts" / "python.exe"
RUN_FEATURES_SCRIPT  = cfg["project_root"] / "run_features.py"
DEVICE = get_device()
# -------------------------
# SUBPROCESS (one definition)
# -------------------------
def run_features_subprocess(participant_id, block_id):
    print("[main] Handing off to venv_feat...")
    result = subprocess.run(
        [str(FEAT_PYTHON), "-u", str(RUN_FEATURES_SCRIPT), participant_id, block_id],
    )
    if result.returncode != 0:
        print("[ERROR] Feature extraction failed")
        sys.exit(1)
    print("[main] Feature extraction complete.")
# -------------------------
# VALIDATION
# -------------------------

def validate_trials(trials):
    for t in trials:
        assert t["stimulus_start"] < t["stimulus_end"]

        if t["questions"]:
            q0 = t["questions"][0]

            assert t["stimulus_end"] <= q0["start"]

            gap = q0["start"] - t["stimulus_end"]

            # expected falloff window
            assert 2.5 <= gap <= 6.0, f"Invalid falloff gap: {gap}"

        for q in t["questions"]:
            assert q["start"] < q["end"]


def validate_labels(labels):
    for l in labels:
        assert l["start"] < l["end"], f"Invalid clip timing: {l}"

        if l["type"] == "stimulus":
            assert "falloff_sec" in l

        if l["type"] == "response":
            assert "question_id" in l


def validate_output_structure(base_path):
    assert base_path.exists()

    required = [
        "stimuli",
        "questions",
        "labels.json",
        "trials.json",
        "metadata.json",
        "sync.json",
    ]

    for r in required:
        assert (base_path / r).exists(), f"Missing: {r}"


# -------------------------
# HELPERS
# -------------------------

def save_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def apply_offset(trials, offset):
    synced = copy.deepcopy(trials)

    for t in synced:
        t["stimulus_start"] += offset
        t["stimulus_end"] += offset

        for q in t["questions"]:
            q["start"] += offset
            q["end"] += offset

    return synced


# -------------------------
# LABEL BUILDER (CORE)
# -------------------------

def build_clip_labels(block, trials):
    labels = []

    for t in trials:
        trial_id = t["trial_id"]

        # -----------------
        # FALL-OFF (CRITICAL)
        # -----------------
        if t["questions"]:
            falloff_end = t["questions"][0]["start"]
        else:
            falloff_end = t["stimulus_end"] + 5.0  # fallback

        falloff_sec = round(falloff_end - t["stimulus_end"], 3)

        # -----------------
        # STIMULUS CLIP
        # -----------------
        labels.append({
            "file": f"stimuli/{trial_id}_stim.mp4",
            "participant": block.participant_id,
            "block": block.block_id,
            "trial_id": trial_id,
            "type": "stimulus",
            "emotion": t["emotion"],
            "is_neutral": t["is_neutral"],
            "start": t["stimulus_start"],
            "end": falloff_end,
            "falloff_sec": falloff_sec
        })

        # -----------------
        # QUESTION CLIPS
        # -----------------
        for q in t["questions"]:
            labels.append({
                "file": f"questions/{trial_id}_{q['question_id']}.mp4",
                "participant": block.participant_id,
                "block": block.block_id,
                "trial_id": trial_id,
                "type": "response",
                "question_id": q["question_id"],
                "question_number": q["question_number"],
                "emotion": t["emotion"],
                "is_neutral": t["is_neutral"],
                "start": q["start"],
                "end": q["end"]
            })

    return labels


# -------------------------
# CORE PIPELINE
# -------------------------

def run_block(participant_id, block_id):
    session = Session(RAW_ROOT, participant_id)
    block = session.get_block(block_id)

    events = block.load_events()

    stimuli = get_event_pairs(events, "stimulus_start", "stimulus_end")
    responses = get_event_pairs(events, "question_start", "question_end")

    trials = build_trials_with_questions(stimuli, responses)

    return block, events, trials


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    participant_id = "AAAD"
    block_id = "B03"

    RUN = {
        "sync": True,
        "clip": True,
        "transcribe": True,
        "features": True,
    }

    base_path = OUTPUT / participant_id / block_id

    if RUN["features"] and not any([RUN["sync"], RUN["clip"], RUN["transcribe"]]):
        run_features_subprocess(participant_id, block_id)   # ← pass IDs not path
        exit()

    block, events, trials = run_block(participant_id, block_id)
    validate_trials(trials)

    sync_info = compute_sync_offset(block, events)
    offset = sync_info["offset"]

    synced_trials = apply_offset(trials, offset)
    validate_trials(synced_trials)

    save_json(sync_info, base_path / "sync.json")
    save_json(synced_trials, base_path / "trials.json")

    metadata = build_metadata(block, trials, events)
    metadata["sync_offset"] = offset
    save_json(metadata, base_path / "metadata.json")

    labels = build_clip_labels(block, synced_trials)
    validate_labels(labels)
    save_json(labels, base_path / "labels.json")

    if RUN["clip"]:
        clip_trials(block, labels, OUTPUT)

    if RUN["transcribe"]:
        model = whisper.load_model("medium")
        labels = transcribe_labels(labels, base_path, model)
        save_json(labels, base_path / "labels.json")

    if RUN["features"]:
        run_features_subprocess(participant_id, block_id)   # ← pass IDs not path

    validate_output_structure(base_path)
    print("Pipeline complete.")