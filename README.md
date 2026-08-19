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

## Phone page

Uploading a garment and placing its anchor points happens on a web page. The
server binds to localhost, so a tunnel puts a public https URL in front of it:

```bash
winget install --id Cloudflare.cloudflared    # or: brew install cloudflared
```

`main.py` starts cloudflared itself and shows the URL it prints as a QR code in
the corner of the mirror — scan it, type the password. With no cloudflared on
PATH there is no QR code and no tunnel; the page is still at
`http://127.0.0.1:8080` for a laptop.

That URL changes every restart, which is why nothing stores it. A free
Cloudflare account and a domain buy a *named* tunnel with a fixed one, and only
the `cloudflared` command in [tunnel.py](tunnel.py) changes.

**`GARMENT_BOT_PASSWORD` in `.env` has to be strong.** Everyone who can see the
QR code has the URL, so the password is the only thing between them and the
mirror. The server allows one guess a second, which a 4-digit PIN survives for
under three hours and a three-word phrase survives for longer than the demo.
