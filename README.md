# magic-mirror

## Setup

Built and tested against Python 3.12. If it is not on the machine,
`uv python install 3.12` first.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The pose model is git-ignored, so a fresh clone does not have it. Download it
into the repo root:

```bash
curl -L -o pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Run

```bash
python main.py    # live webcam pose tracking, press q to quit
```
