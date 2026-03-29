from pathlib import Path
import pandas as pd

class Block:
    def __init__(self, participant_id: str, block_id: str, block_path: Path):
        self.participant_id = participant_id
        self.block_id = block_id
        self.block_path = block_path

        self.video_path = block_path / f"{participant_id}_{block_id}.mp4"
        self.csv_path   = block_path / f"{participant_id}_{block_id}.csv"
        self.xml_path   = block_path / f"{participant_id}_{block_id}.xml"

        self.events = None  # loaded later

    def load_events(self):
        df = pd.read_csv(self.csv_path)

        # enforce structure
        df.columns = ["t", "event", "detail"]

        # sort just in case
        df = df.sort_values("t").reset_index(drop=True)

        self.events = df
        return df
    
class Session:
    def __init__(self, raw_root: Path, participant_id: str):
        self.raw_root = Path(raw_root)
        self.participant_id = participant_id

        self.base = self.raw_root / participant_id

    def get_block(self, block_id: str) -> Block:
        block_path = self.base / block_id

        if not block_path.exists():
            raise FileNotFoundError(f"Block folder not found: {block_path}")

        return Block(self.participant_id, block_id, block_path)
    
def get_events(df, event_name):
    return df[df["event"] == event_name]


def get_event_pairs(df, start_event, end_event):
    starts = df[df["event"] == start_event].reset_index(drop=True)
    ends   = df[df["event"] == end_event].reset_index(drop=True)

    assert len(starts) == len(ends), "Mismatch start/end events"

    pairs = []
    for i in range(len(starts)):
        pairs.append({
            "start": starts.loc[i, "t"],
            "end": ends.loc[i, "t"],
            "detail": starts.loc[i, "detail"]
        })

    return pairs

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