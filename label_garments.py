"""
Ground truth for the anchor model, made by correcting rather than clicking.

Every photo opens with its 7 anchors already placed - from a sidecar if one was
saved before, otherwise from auto_anchors - so labelling a garment is dragging
the two or three that are wrong instead of placing all seven. What it writes is
the same sidecar the mirror loads, so a labelled photo is immediately wearable
and immediately scorable.

Usage:
    python label_garments.py [folder]     # defaults to dataset/

Controls:
    drag - move the nearest anchor
    r    - back to the automatic guess
    n    - save and move on
    k    - skip this photo, saving nothing
    q    - quit
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import auto_anchors
from calibrate import _composite_on_checkerboard, POINT_COLORS
from garment_overlay import POINT_NAMES, SEGMENT_REQUIRED_POINTS

DATASET_DIR = "dataset"
VIEW_HEIGHT = 900   # photos arrive around 1500px tall and the window has to fit a screen
GRAB_RADIUS = 20    # in view pixels - how near the cursor must be to pick an anchor up


def prefill(png, rgba):
    """Where to start: a sidecar saved earlier, else what the heuristics guess."""
    sidecar = png.with_suffix(".anchors.json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    return auto_anchors.top_anchors(rgba)


def bones(segments):
    """Every pair of anchors a segment joins: a triangle for the torso, one line for a limb."""
    for segment in segments:
        names = SEGMENT_REQUIRED_POINTS[segment]
        for i, start in enumerate(names):
            for end in names[i + 1:]:
                yield start, end


def nearest_anchor(anchors, x, y, scale):
    """The anchor under the cursor, or None when the cursor is near none of them."""
    # Nearest rather than first within reach: on a sleeveless garment the two
    # elbows can sit close enough together that first-match grabs the wrong one.
    reach = {name: np.hypot(anchors[name][0] * scale - x, anchors[name][1] * scale - y)
             for name in POINT_NAMES}
    closest = min(reach, key=reach.get)
    return closest if reach[closest] <= GRAB_RADIUS else None


def draw(base, anchors, scale, held):
    canvas = base.copy()
    for start, end in bones(anchors["segments"]):
        cv2.line(canvas, tuple(int(anchors[start][i] * scale) for i in (0, 1)),
                 tuple(int(anchors[end][i] * scale) for i in (0, 1)), (255, 255, 255), 1)
    for name, color in zip(POINT_NAMES, POINT_COLORS):
        x, y = int(anchors[name][0] * scale), int(anchors[name][1] * scale)
        cv2.circle(canvas, (x, y), 9 if name == held else 6, color, -1)
        cv2.putText(canvas, name, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def label(png, window):
    """One photo. Returns ("save", anchors), ("skip", None) or ("quit", None)."""
    rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        print(f"{png.name}: no alpha channel - run it through bg_remove first")
        return "skip", None

    guess = prefill(png, rgba)
    if guess is None:
        print(f"{png.name}: nothing to measure in this cutout")
        return "skip", None

    anchors = json.loads(json.dumps(guess))   # a copy, so `r` still has the original
    scale = min(1.0, VIEW_HEIGHT / rgba.shape[0])
    base = cv2.resize(_composite_on_checkerboard(rgba), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)
    held = None

    def on_mouse(event, x, y, flags, userdata):
        nonlocal held, anchors
        if event == cv2.EVENT_LBUTTONDOWN:
            held = nearest_anchor(anchors, x, y, scale)
        elif event == cv2.EVENT_LBUTTONUP:
            held = None
        elif event == cv2.EVENT_MOUSEMOVE and held is not None:
            anchors[held] = [int(x / scale), int(y / scale)]

    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, draw(base, anchors, scale, held))
        key = cv2.waitKey(20) & 0xFF
        if key == ord('n'):
            return "save", anchors
        if key == ord('k'):
            return "skip", None
        if key == ord('q'):
            return "quit", None
        if key == ord('r'):
            anchors = json.loads(json.dumps(guess))


def main(directory):
    photos = sorted(Path(directory).glob("*.png"))
    if not photos:
        sys.exit(f"No .png cutouts in {directory}/ - put background-removed garments there first.")

    window = "label garments - drag to correct, n=next k=skip r=reset q=quit"
    cv2.namedWindow(window)
    saved = 0
    for i, png in enumerate(photos, 1):
        cv2.setWindowTitle(window, f"{png.name}  ({i}/{len(photos)}, {saved} saved)")
        action, anchors = label(png, window)
        if action == "quit":
            break
        if action == "save":
            with open(png.with_suffix(".anchors.json"), "w") as f:
                json.dump(anchors, f, indent=2)
            saved += 1

    cv2.destroyAllWindows()
    print(f"{saved} of {len(photos)} labelled in {directory}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR)
