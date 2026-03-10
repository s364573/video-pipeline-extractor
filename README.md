# video_stimulus_syncer

Utilities for:
- syncing a recorded video stream to experiment timestamps using an audio beep
- clipping stimulus/question segments from the source MP4
- extracting facial features from clipped videos with `py-feat`

## Python version

Use Python `3.11` (see `.python-version`).

## System requirement

`ffmpeg` must be installed and available on `PATH`.

macOS (Homebrew):

```bash
brew install ffmpeg
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data layout

`read_csv.py` looks for participant files in:

1. `PARTICIPANTS_DIR` (if set), else
2. local `participant/`, else
3. legacy `~/01_MASTER/Data/Participants` (if present)

Expected files are matched by participant/block in filename, with extensions:
- `.mp4`
- `.wav`
- `.csv`

## Run

### 1) Sync quick check (`main.py`)

```bash
python main.py
```

### 2) Clip block videos/questions (`test.py`)

Edit the example call at the bottom of `test.py` if needed:

```python
process_block("RRRR", "03")
```

Then run:

```bash
python test.py
```

### 3) Extract features (`feature.py`)

Edit the example call at the bottom if needed:

```python
extract_features_standalone("RRRR", "00")
```

Then run:

```bash
python feature.py
```

## Notes

- The `requirements.txt` now contains only direct project dependencies.
- Transitive dependencies are resolved by pip automatically.
