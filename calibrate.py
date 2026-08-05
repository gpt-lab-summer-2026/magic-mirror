"""
One-time calibration tool.

Click 3 points on a background-removed garment image, in this exact order:
  1. Left shoulder  (the wearer's left shoulder)
  2. Right shoulder (the wearer's right shoulder)
  3. Hip center      (roughly where the hips would sit, centered horizontally)

These three points become the "src" side of the affine transform used every
frame to warp the garment onto the tracked body. The order matters — it must
match the order the body-side triangle is built in (see garment_overlay.py).

Saves a sidecar JSON next to the image: <name>.anchors.json

Usage:
    python calibrate.py test-images-output/bg_white_top.out.png

Controls:
    click   - place the next point
    u       - undo last point
    s       - save (once all 3 points are placed)
    q       - quit without saving
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

POINT_NAMES = ["left_shoulder", "right_shoulder", "hip_center"]
POINT_COLORS = [(0, 200, 255), (255, 200, 0), (0, 255, 0)]  # BGR, one per point


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
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))

    window_name = "Calibration - click left shoulder, right shoulder, hip center (u=undo, s=save, q=quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = display_base.copy()
        for i, (px, py) in enumerate(points):
            color = POINT_COLORS[i]
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.putText(frame, POINT_NAMES[i], (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        next_label = POINT_NAMES[len(points)] if len(points) < 3 else "all points set - press s to save"
        cv2.putText(frame, f"click: {next_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and points:
            points.pop()
        elif key == ord('s'):
            if len(points) == 3:
                break
            print("Need exactly 3 points before saving.")
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
