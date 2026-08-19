"""
The anchors an upload starts from, and the session that carries them.

auto_anchors reads a top off its own silhouette; nothing reads a pair of
trousers, and a shirt it cannot measure still has to reach the phone somehow.
So every upload gets a draft sidecar - measured where that works, spread over
the garment's bounding box where it does not - and whoever sent the photo drags
it into place on the anchor page. The sessions live here rather than in
telegram_bot.py because anchor_server.py has to reach the same ones.
"""
import json
import secrets
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import auto_anchors
import garment_overlay
import garment_overlay_bottom
from auto_anchors import ALPHA_CUTOFF, BASE_OCCLUDERS
from garment_overlay import POINT_NAMES, CORE_POINT_COUNT as CORE_POINT_COUNT_TOP
from garment_overlay_bottom import POINT_NAMES_BOTTOM, CORE_POINT_COUNT as CORE_POINT_COUNT_BOTTOM
from garment_types import RIG_BY_CATEGORY

# Cycled by point index - long enough for the longer of the two point sets
# (bottoms, at 8), and wraps via modulo wherever it is read. It lives here
# rather than in calibrate.py, which now imports it, so that the calibration
# window, the preview photo and the anchor page all colour a point alike.
POINT_COLORS = [
    (0, 200, 255),    # orange
    (255, 200, 0),    # cyan
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (0, 128, 255),    # amber
    (255, 255, 0),    # yellow
    (128, 0, 255),    # purple
    (0, 0, 255),      # red
]

# The two keys in a sidecar that are not coordinates.
RIG_KEYS = ("segments", "occluders")

# Which points a rig collects, how many of them it cannot do without, and what
# each of its segments needs - the very map garment_rig.load_calibration checks
# a sidecar against. Keyed by rig, so garment_types.py stays the only list of
# categories.
RIGS = {
    "top": (POINT_NAMES, CORE_POINT_COUNT_TOP, garment_overlay.SEGMENT_REQUIRED_POINTS),
    "bottom": (POINT_NAMES_BOTTOM, CORE_POINT_COUNT_BOTTOM, garment_overlay_bottom.SEGMENT_REQUIRED_POINTS),
}

INSET = 0.1      # of the box width: its very corner is a cuff or a pocket, never a seam
HIP_DROP = 0.6   # of the box height, for a top no finder could measure


def decode(png: bytes):
    """PNG bytes as the RGBA array every function here measures."""
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


def points_for(category: str):
    """The category's point names, and how many of them are core."""
    names, core, _ = RIGS[RIG_BY_CATEGORY[category]]
    return names, core


def points_of(sidecar: dict):
    """A sidecar's coordinates alone, in the order it lists them."""
    return {name: xy for name, xy in sidecar.items() if name not in RIG_KEYS}


def _box(rgba):
    """The garment's own bounding box as (x, y, width, height)."""
    opaque = rgba[:, :, 3] > ALPHA_CUTOFF
    cols, rows = np.flatnonzero(opaque.any(0)), np.flatnonzero(opaque.any(1))
    if not len(cols):
        raise ValueError("that cutout came out empty - nothing left to place points on")
    return cols[0], rows[0], cols[-1] - cols[0], rows[-1] - rows[0]


def bottom_layout(rgba, point_names):
    """Every bottom point spread over the bounding box, as somewhere to drag from.

    Left and right are the wearer's, so the left points sit on the image's
    right - the convention auto_anchors.shoulder_points sets out and every
    sidecar in garments/ follows.
    """
    x, y, w, h = _box(rgba)
    left, right, middle = x + (1 - INSET) * w, x + INSET * w, x + w / 2
    spread = {
        "hip_center": (middle, y),
        "left_waist": (left, y),
        "right_waist": (right, y),
        "crotch": (middle, y + 0.45 * h),
        "left_knee": (left, y + 0.70 * h),
        "right_knee": (right, y + 0.70 * h),
        "left_ankle": (left, y + 0.95 * h),
        "right_ankle": (right, y + 0.95 * h),
    }
    return {name: spread[name] for name in point_names}


def top_layout(rgba):
    """The same spread for a top the silhouette reader could not measure: shoulders
    at the top corners, arms hanging off them the way auto_anchors hangs them."""
    x, y, w, h = _box(rgba)
    spread = {
        "left_shoulder": (x + (1 - INSET) * w, y),
        "right_shoulder": (x + INSET * w, y),
        "hip_center": (x + w / 2, y + HIP_DROP * h),
    }
    width = spread["left_shoulder"][0] - spread["right_shoulder"][0]
    for side in ("left", "right"):
        shoulder = spread[f"{side}_shoulder"]
        wrist = auto_anchors.hanging_wrist(shoulder, width, side)
        spread[f"{side}_elbow"] = ((shoulder[0] + wrist[0]) / 2, (shoulder[1] + wrist[1]) / 2)
        spread[f"{side}_wrist"] = wrist
    return spread


def _sidecar(points, point_names, segments, occluders):
    """One sidecar shape for every category: whole pixels, in point_names order -
    so a colour index means the same thing in the file, on the preview and on the
    page - then the two keys garment_rig.load_calibration insists on."""
    sidecar = {name: [int(points[name][0]), int(points[name][1])]
               for name in point_names if name in points}
    sidecar["segments"] = segments
    sidecar["occluders"] = occluders
    return sidecar


def initial_sidecar(rgba, category: str):
    """The draft to start from, and whether it was measured rather than guessed.

    Only a measured top is trusted enough to publish unseen; a spread is a
    starting position for the page, never a garment.
    """
    if category == "shirt":
        measured = auto_anchors.top_anchors(rgba)
        if measured is not None:
            return _sidecar(measured, POINT_NAMES, measured["segments"], measured["occluders"]), True
        # No sleeves were found because nothing was measured at all, so repaint
        # the arms - auto_anchors' own rule for a top that turned out sleeveless.
        return _sidecar(top_layout(rgba), POINT_NAMES, ["torso"], BASE_OCCLUDERS + ["arms"]), False

    if category == "pants":
        return _sidecar(bottom_layout(rgba, POINT_NAMES_BOTTOM), POINT_NAMES_BOTTOM,
                        ["seat", "left_upper_leg", "right_upper_leg",
                         "left_lower_leg", "right_lower_leg"], ["hands", "shoes"]), False

    if category == "skirt":
        # Core points only: a skirt is one sheet of fabric with no leg seam, and
        # garment_library reads exactly this segment list as "warp it as a sheet".
        core = POINT_NAMES_BOTTOM[:CORE_POINT_COUNT_BOTTOM]
        return _sidecar(bottom_layout(rgba, core), core, ["seat"], ["shoes"]), False

    raise ValueError(f"no such category {category!r} - pick one of {', '.join(RIG_BY_CATEGORY)}")


def composite_on_checkerboard(rgba, square=16):
    """Composite onto a checkerboard so transparent edges are actually visible."""
    h, w = rgba.shape[:2]
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            shade = 60 if ((x // square) + (y // square)) % 2 == 0 else 90
            board[y:y + square, x:x + square] = (shade, shade, shade)

    if rgba.shape[2] == 4:
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        composited = rgb * alpha + board.astype(np.float32) * (1 - alpha)
        return composited.astype(np.uint8)
    return rgba[:, :, :3]


def render_preview(rgba, anchors: dict) -> bytes:
    """The draft drawn on the cutout as PNG bytes: calibrate.py's window, for
    someone holding a phone."""
    frame = composite_on_checkerboard(rgba)
    for i, (name, (px, py)) in enumerate(points_of(anchors).items()):
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.putText(frame, name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.imencode(".png", frame)[1].tobytes()


def web_colors():
    """POINT_COLORS as CSS hex - the page draws the same dots, in RGB."""
    return [f"#{r:02x}{g:02x}{b:02x}" for b, g, r in POINT_COLORS]


def validate(data: str, draft: dict, category: str, width: int, height: int) -> dict:
    """The page's JSON as a sidecar, or ValueError carrying one line to reply with.

    Anything at all can arrive here: the string comes off the network, and the
    page it comes from is a URL anyone can open.
    """
    try:
        sent = json.loads(data)
    except json.JSONDecodeError:
        sent = None
    if not isinstance(sent, dict):
        raise ValueError("that didn't look like anchors - open the page again")

    names, core, segment_needs = RIGS[RIG_BY_CATEGORY[category]]
    unknown = [name for name in sent if name not in names]
    if unknown:
        raise ValueError(f"a {category} has no {', '.join(unknown)}")
    missing = [name for name in names[:core] if name not in sent]
    if missing:
        raise ValueError(f"still to place: {', '.join(missing)}")

    for name, value in sent.items():
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) for v in value):
            raise ValueError(f"{name} is not a pair of whole numbers")
        if not (0 <= value[0] < width and 0 <= value[1] < height):
            raise ValueError(f"{name} landed outside the image")

    # A segment whose points were switched off is one this garment can no longer
    # wear: the rule load_calibration enforces at load time, applied here where
    # there is still somebody to tell. A top left with no sleeve points comes out
    # of it as ["torso"], which is what makes it sleeveless - so its bare arms
    # get repainted, exactly as auto_anchors does for a sleeveless measurement.
    segments = [s for s in draft["segments"] if all(p in sent for p in segment_needs[s])]
    occluders = draft["occluders"]
    if category == "shirt" and segments == ["torso"]:
        occluders = BASE_OCCLUDERS + ["arms"]
    return _sidecar(sent, names, segments, occluders)


# One session per chat, the token inside it: a second upload replaces the whole
# entry, so the token it held stops resolving the moment it is no longer current.
_sessions = {}


def park_photo(chat_id, png: bytes):
    """A photo arrives before its category, so it waits here for the button press."""
    _sessions[chat_id] = {"photo": png}


def take_photo(chat_id):
    """The parked photo, once: a category press consumes it, so pressing a button
    twice asks for a photo again."""
    session = _sessions.pop(chat_id, None)
    return session.get("photo") if session else None


def new_session(chat_id, cutout: bytes, sidecar: dict, category: str) -> str:
    """Park a draft for the anchor page and return the token that reaches it."""
    height, width = decode(cutout).shape[:2]
    token = secrets.token_urlsafe(24)
    _sessions[chat_id] = {"token": token, "cutout": cutout, "sidecar": sidecar,
                          "category": category, "width": width, "height": height}
    return token


def get_session(token: str):
    """The session that token belongs to, or None - the server's whole lookup."""
    for session in _sessions.values():
        if session.get("token") == token:
            return session
    return None


def get_chat_session(chat_id):
    """The chat's open draft, or None - a photo waiting for a category is not one."""
    session = _sessions.get(chat_id)
    return session if session and "token" in session else None


def end_session(chat_id):
    _sessions.pop(chat_id, None)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python anchor_session.py cutout.png [{'|'.join(RIG_BY_CATEGORY)}]")

    path, category = Path(sys.argv[1]), sys.argv[2]
    rgba = decode(path.read_bytes())
    sidecar, trusted = initial_sidecar(rgba, category)
    print(json.dumps(sidecar, indent=2))
    print(f"trusted: {trusted}")

    # Scratch, never beside the garment: this is a look at a draft, not a calibration.
    preview = Path(tempfile.gettempdir()) / f"{path.stem}.preview.png"
    preview.write_bytes(render_preview(rgba, sidecar))
    print(f"preview: {preview}")
