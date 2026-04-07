import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import copy
from data.loader import Session
from data.parser import get_event_pairs
from data.trials import build_trials_with_questions, build_metadata
from pipeline.sync import compute_sync_offset
from pipeline.clip import clip_trials
from pipeline.transcribe import transcribe_labels, load_model as load_whisper_model
from pipeline.features import extract_features
from utils.config_loader import load_config

# -------------------------
# PATHS FROM CONFIG
# -------------------------
cfg = load_config()
RAW_ROOT = cfg["raw_root"]
OUTPUT   = cfg["output_root"]


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
    for r in ["stimuli", "questions", "labels.json", "trials.json", "metadata.json", "sync.json"]:
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
# LABEL BUILDER
# -------------------------
def build_clip_labels(block, trials):
    labels = []
    for t in trials:
        trial_id = t["trial_id"]
        falloff_end = t["questions"][0]["start"] if t["questions"] else t["stimulus_end"] + 5.0
        falloff_sec = round(falloff_end - t["stimulus_end"], 3)

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


def process_block(participant_id, block_id, RUN=None):
    """
    Run the full pipeline for one participant/block.
    RUN dict controls which stages are active.
    """
    if RUN is None:
        RUN = {"sync": True, "clip": True, "transcribe": True, "features": True}

    base_path = OUTPUT / participant_id / block_id

    # features-only shortcut
    if RUN["features"] and not any([RUN["sync"], RUN["clip"], RUN["transcribe"]]):
        extract_features(base_path)
        return

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
        model = load_whisper_model()
        labels = transcribe_labels(labels, base_path, model)
        save_json(labels, base_path / "labels.json")

    if RUN["features"]:
        extract_features(base_path)

    validate_output_structure(base_path)
    print(f"[DONE] {participant_id} / {block_id}")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    process_block(
        participant_id="AAAD",
        block_id="B03",
        RUN={"sync": True, "clip": True, "transcribe": True, "features": True},
    )
