from pathlib import Path
import pandas as pd


def _resolve_block_file(block_path: Path, filename: str) -> Path:
    direct_path = block_path / filename
    if direct_path.exists():
        return direct_path

    target_name = filename.lower()
    for candidate in block_path.iterdir():
        if candidate.name.lower() == target_name:
            return candidate

    return direct_path


def _block_exists(block_path: Path, participant_id: str, block_id: str) -> bool:
    if not block_path.exists():
        return False

    expected = [
        f"{participant_id}_{block_id}.csv",
        f"{participant_id}_{block_id}.mp4",
        f"{participant_id}_{block_id}.wav",
        f"{participant_id}_{block_id}.xml",
    ]
    return any(_resolve_block_file(block_path, name).exists() for name in expected)


class Block:
    def __init__(self, participant_id: str, block_id: str, block_path: Path):
        self.participant_id = participant_id
        self.block_id = block_id
        self.block_path = block_path

        self.video_path = _resolve_block_file(
            block_path, f"{participant_id}_{block_id}.mp4"
        )
        self.csv_path = _resolve_block_file(
            block_path, f"{participant_id}_{block_id}.csv"
        )
        self.xml_path = _resolve_block_file(
            block_path, f"{participant_id}_{block_id}.xml"
        )
        self.wav_path = _resolve_block_file(
            block_path, f"{participant_id}_{block_id}.wav"
        )

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
        nested_block_path = self.base / block_id

        if _block_exists(nested_block_path, self.participant_id, block_id):
            block_path = nested_block_path
        elif _block_exists(self.base, self.participant_id, block_id):
            block_path = self.base
        else:
            raise FileNotFoundError(
                f"Block files not found for participant={self.participant_id} block={block_id}"
            )

        return Block(self.participant_id, block_id, block_path)
