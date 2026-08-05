"""
One-time calibration tool.

Click points on a background-removed garment image, in this exact order
(defined once, in garment_overlay.POINT_NAMES, so this file and the runtime
overlay can never disagree about ordering):
  1. Left shoulder
  2. Right shoulder
  3. Hip center   (roughly where the hips would sit, centered horizontally)
  4. Left elbow
  5. Right elbow
  6. Left wrist
  7. Right wrist

These become the "src" side of the affine fit used every frame to warp the
garment onto the tracked body. At runtime, whichever of these points has a
currently-visible match on the body (shoulders + hip-center are always
required; elbows/wrists are used opportunistically) get fed into a
least-squares affine solve.

Saves a sidecar JSON next to the image: <name>.anchors.json

Usage:
    python calibrate.py test-images-output/bg_white_top.out.png

Controls:
    click   - place the next point
    u       - undo last point
    s       - save (once all points are placed)
    q       - quit without saving
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

from garment_overlay import POINT_NAMES

POINT_COLORS = [
    (0, 200, 255),    # left_shoulder - orange
    (255, 200, 0),    # right_shoulder - cyan
    (0, 255, 0),      # hip_center - green
    (255, 0, 255),    # left_elbow - magenta
    (0, 128, 255),    # right_elbow - amber
    (255, 255, 0),    # left_wrist - yellow
    (128, 0, 255),    # right_wrist - purple
]


def _composite_on_checkerboard(rgba, square=16):
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


def calibrate(image_path: str) -> dict:
    rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    display_base = _composite_on_checkerboard(rgba)
    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(POINT_NAMES):
            points.append((x, y))

    window_name = "Calibration - click each point in order (u=undo, s=save, q=quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = display_base.copy()
        for i, (px, py) in enumerate(points):
            color = POINT_COLORS[i]
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.putText(frame, POINT_NAMES[i], (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        next_label = POINT_NAMES[len(points)] if len(points) < len(POINT_NAMES) else "all points set - press s to save"
        cv2.putText(frame, f"click: {next_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and points:
            points.pop()
        elif key == ord('s'):
            if len(points) == len(POINT_NAMES):
                break
            print(f"Need all {len(POINT_NAMES)} points before saving ({len(points)} placed so far).")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit("Calibration cancelled.")

    cv2.destroyAllWindows()

    return {name: list(pt) for name, pt in zip(POINT_NAMES, points)}


def save_anchors(image_path: str, anchors: dict) -> str:
    out_path = str(Path(image_path).with_suffix("")) + ".anchors.json"
    with open(out_path, "w") as f:
        json.dump(anchors, f, indent=2)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python calibrate.py path/to/garment.out.png")

    img_path = sys.argv[1]
    anchors = calibrate(img_path)
    out_path = save_anchors(img_path, anchors)
    print(f"Saved anchors to: {out_path}")
    print(anchors)
