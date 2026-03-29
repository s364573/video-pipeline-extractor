def parse_question_detail(detail):
    # B01_01_surprise_Q03
    base, q = detail.rsplit("_Q", 1)
    return base, f"Q{q}"

def extract_emotion(trial_id):
    if trial_id.startswith("NF"):
        return "neutral"
    return trial_id.split("_")[-1].lower()

def build_trials_with_questions(stimuli, responses):
    trials = {}

    # 1. Initialize from stimuli
    for stim in stimuli:
        trial_id = stim["detail"]

        trials[trial_id] = {
            "trial_id": trial_id,
            "stimulus_start": stim["start"],
            "stimulus_end": stim["end"],
            "is_neutral": trial_id.startswith("NF"),
            "emotion": extract_emotion(trial_id),
            "questions": []
            }

    # 2. Attach responses
    for r in responses:
        trial_id, q_id = parse_question_detail(r["detail"])

        if trial_id not in trials:
            print(f"WARNING: response without stimulus: {r['detail']}")
            continue

        trials[trial_id]["questions"].append({
            "question_id": q_id,
            "start": r["start"],
            "end": r["end"]
        })

    # 3. Sort questions + ADD INDEX HERE
    for t in trials.values():
        t["questions"] = sorted(t["questions"], key=lambda x: x["start"])

        for i, q in enumerate(t["questions"]):
            q["index"] = i

            # optional upgrade
            q["question_number"] = int(q["question_id"][1:])  # Q03 → 3

    return list(trials.values())

def build_metadata(block, trials, events):
    return {
        "participant_id": block.participant_id,
        "block_id": block.block_id,
        "n_trials": len(trials),
        "n_questions": sum(len(t["questions"]) for t in trials),
        "n_neutral_trials": sum(t["is_neutral"] for t in trials),
        "n_emotion_trials": sum(not t["is_neutral"] for t in trials),
        "csv_beep_time": float(
            events[events["event"] == "beep_start"]["t"].iloc[0]
        )
    }