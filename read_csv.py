import csv
import os
from pathlib import Path


def extract_beep_times(filename):
    beep_times = []

    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["event"] == "beep_start" or row["event"] == "beep_end":
                beep_times.append(row["t"])

    return beep_times


# Default to local project data; allow override with PARTICIPANTS_DIR.
_default_folder = Path("participant")
_legacy_folder = Path.home() / "01_MASTER" / "Data" / "Participants"
folder = Path(os.environ.get("PARTICIPANTS_DIR", _default_folder))
if not folder.exists() and _legacy_folder.exists():
    folder = _legacy_folder


def load_participant_files(participant, block):
    # Initialize the container for our specific types
    found_files = {"mp4": None, "wav": None, "csv": None}

    # Define the set of extensions we are looking for
    valid_extensions = {".mp4", ".wav", ".csv"}
    participant_folder = folder / participant

    # Single loop through the directory
    for file in participant_folder.rglob("*"):
        ext = file.suffix.lower()
        if file.is_dir() or ext not in valid_extensions:
            continue

        name = file.stem
        if "_" not in name:
            continue

        pid, rest = name.split("_", 1)
        blockid = rest.split("-", 1)[0] if "-" in rest else rest

        # Check if the file belongs to the right person and block
        if pid == participant and blockid == block:
            # Map the extension (stripping the dot) to the file path
            key = ext.replace(".", "")
            found_files[key] = file

    return found_files
