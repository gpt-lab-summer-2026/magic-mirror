# magic-mirror

## Setup

Built and tested against Python 3.12. If it is not on the machine,
`uv python install 3.12` first.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On a machine with no NVIDIA GPU, install torch from the CPU index first — the
default Linux wheel carries a ~2-3 GB CUDA runtime that will never be used:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
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

Keys: `<-` `->` or `1-9` to change garment, `d` debug HUD, `l` relight,
`s` dump a frame, `f` fullscreen, `q` quit.

The demo camera is picked by the by-id link in [config.py](config.py); anywhere
that link does not resolve it falls back to `CAMERA_INDEX`.

The human parser needs a GPU — it runs at ~1.5 s per frame on CPU and starves
the render loop, so without CUDA it is skipped. Everything else works; garments
just draw over hands and bare arms instead of behind them.
