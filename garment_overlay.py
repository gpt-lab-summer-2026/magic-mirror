"""
Loading a calibrated garment and overlaying it on a camera frame each loop
iteration using a thin-plate spline (TPS) warp driven by MediaPipe pose
landmarks (shoulders, hip-center, elbows, wrists).

Requires opencv-contrib-python (cv2.createThinPlateSplineShapeTransformer
lives in the contrib "shape" module, not plain opencv-python). If you have
plain opencv-python installed, you'll likely need to uninstall it first —
having both installed at once is a common source of import conflicts,
since they both provide the "cv2" module:
    pip uninstall opencv-python
    pip install opencv-contrib-python --break-system-packages
"""
import json
from pathlib import Path

import cv2
import numpy as np

# The full set of anchor points calibrate.py collects, in the order it
# collects them in. Single source of truth — calibrate.py imports this
# directly so the two files can never drift out of sync on ordering.
POINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "hip_center",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

# MediaPipe Pose landmark indices we care about.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# name -> landmark index, for the points that are optional at runtime (an
# arm can swing out of frame without blocking the overlay entirely).
OPTIONAL_LANDMARKS = {
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
}

MIN_VISIBILITY = 0.5

# Fallback used when hip landmarks aren't visible — very common, since most
# webcam mirror setups frame from around the chest or waist up and hips
# never make it into the shot. Vertical distance from the shoulder midpoint
# down to the hip midpoint, expressed as a multiple of shoulder width.
HIP_OFFSET_RATIO = 1.2

# TPS regularization: 0 = exact interpolation (the warp is forced through
# every tracked point exactly, with zero slack — a single noisy point will
# visibly warp the whole garment to match it). Raise this above 0 to trade
# exactness for tolerance to jittery points, if that turns out to matter
# in practice once this is running.
TPS_REGULARIZATION = 0.0

# How far past the tracked point cloud's bounding box to still compute and
# draw the warp, as a multiple of shoulder width. Keeps the (fairly
# expensive) per-pixel TPS evaluation limited to a sensible region around
# the body instead of the full frame, and avoids TPS's tendency to
# extrapolate wildly far from the data it was fit on.
PADDING_RATIO = 0.75

_last_debug_message = None


def _debug_log(message: str) -> None:
    """Print only when the tracking state changes, so this doesn't spam the console every frame."""
    global _last_debug_message
    if message != _last_debug_message:
        print(f"[garment_overlay] {message}", flush=True)
        _last_debug_message = message


class Garment:
    """A background-removed garment image plus its calibrated anchor points."""

    def __init__(self, image_path: str):
        rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Could not read garment image: {image_path}")
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Garment image must have an alpha channel (RGBA): {image_path}")
        self.rgba = rgba

        anchors_path = str(Path(image_path).with_suffix("")) + ".anchors.json"
        if not Path(anchors_path).exists():
            raise FileNotFoundError(
                f"No calibration found for {image_path}. "
                f"Run: python calibrate.py {image_path}"
            )
        with open(anchors_path) as f:
            raw_anchors = json.load(f)

        missing = [name for name in POINT_NAMES if name not in raw_anchors]
        if missing:
            raise ValueError(
                f"Calibration file for {image_path} is missing {missing}. "
                f"Re-run: python calibrate.py {image_path}"
            )

        self.anchors = {name: np.float32(raw_anchors[name]) for name in POINT_NAMES}


class LandmarkSmoother:
    """
    Exponential moving average over a named set of points, to reduce
    jitter. Handles the point set changing size/composition frame to frame
    (e.g. a wrist appearing or disappearing) by smoothing each name
    independently against its own history.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._smoothed = {}  # name -> np.array([x, y])

    def update(self, named_points: dict) -> dict:
        result = {}
        for name, pt in named_points.items():
            pt = np.asarray(pt, dtype=np.float32)
            if name in self._smoothed:
                self._smoothed[name] = self.alpha * pt + (1 - self.alpha) * self._smoothed[name]
            else:
                self._smoothed[name] = pt
            result[name] = self._smoothed[name]
        return result

    def reset(self):
        self._smoothed = {}


def get_body_points(pose_landmarks, frame_width, frame_height):
    """
    Build a dict of currently-tracked named body points, in pixel
    coordinates. Shoulders are required (returns None if either is missing
    or low-confidence). Hip-center always comes back, real if visible,
    synthesized from the shoulders otherwise. Elbows and wrists are
    opportunistic: each is included only if that specific landmark is
    visible this frame.
    """
    lm = pose_landmarks

    if max(LEFT_SHOULDER, RIGHT_SHOULDER) >= len(lm):
        _debug_log("pose result has too few landmarks")
        return None

    if lm[LEFT_SHOULDER].visibility < MIN_VISIBILITY or lm[RIGHT_SHOULDER].visibility < MIN_VISIBILITY:
        _debug_log("shoulders not visible enough - face the camera / adjust lighting")
        return None

    def to_px(landmark):
        return np.array([landmark.x * frame_width, landmark.y * frame_height], dtype=np.float32)

    named_points = {
        "left_shoulder": to_px(lm[LEFT_SHOULDER]),
        "right_shoulder": to_px(lm[RIGHT_SHOULDER]),
    }

    hips_visible = (
        max(LEFT_HIP, RIGHT_HIP) < len(lm)
        and lm[LEFT_HIP].visibility >= MIN_VISIBILITY
        and lm[RIGHT_HIP].visibility >= MIN_VISIBILITY
    )
    if hips_visible:
        named_points["hip_center"] = (to_px(lm[LEFT_HIP]) + to_px(lm[RIGHT_HIP])) / 2.0
        _debug_log("tracking with real hip landmarks")
    else:
        shoulder_mid = (named_points["left_shoulder"] + named_points["right_shoulder"]) / 2.0
        shoulder_width = np.linalg.norm(named_points["right_shoulder"] - named_points["left_shoulder"])
        named_points["hip_center"] = shoulder_mid + np.array(
            [0.0, shoulder_width * HIP_OFFSET_RATIO], dtype=np.float32
        )
        _debug_log("hips out of frame - using synthetic hip-center estimate")

    for name, idx in OPTIONAL_LANDMARKS.items():
        if idx < len(lm) and lm[idx].visibility >= MIN_VISIBILITY:
            named_points[name] = to_px(lm[idx])

    return named_points


def warp_and_blend(frame_bgr: np.ndarray, garment: Garment, named_dst_points: dict) -> np.ndarray:
    """
    Fit a thin-plate spline from whichever garment anchors currently have a
    matching tracked body point, warp the garment with it, and alpha-blend
    onto frame_bgr within a padded region around the tracked points.
    Returns a new frame; does not mutate frame_bgr in place. Returns
    frame_bgr unchanged if fewer than 3 correspondences are available.
    """
    names = [name for name in POINT_NAMES if name in named_dst_points and name in garment.anchors]
    if len(names) < 3:
        return frame_bgr

    src_pts = np.float32([garment.anchors[name] for name in names]).reshape(1, -1, 2)
    dst_pts = np.float32([named_dst_points[name] for name in names]).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(names))]

    tps = cv2.createThinPlateSplineShapeTransformer(TPS_REGULARIZATION)
    # transformingShape=dst_pts (frame space), targetShape=src_pts (garment
    # space) — so applyTransformation() on frame-space query points below
    # returns the corresponding garment-space coordinates, which is exactly
    # the "where do I sample from" mapping cv2.remap needs.
    tps.estimateTransformation(dst_pts, src_pts, matches)

    frame_h, frame_w = frame_bgr.shape[:2]

    xs, ys = dst_pts[0, :, 0], dst_pts[0, :, 1]
    shoulder_width = np.linalg.norm(
        named_dst_points["right_shoulder"] - named_dst_points["left_shoulder"]
    )
    pad = max(shoulder_width * PADDING_RATIO, 1.0)

    x0 = max(int(np.floor(xs.min() - pad)), 0)
    y0 = max(int(np.floor(ys.min() - pad)), 0)
    x1 = min(int(np.ceil(xs.max() + pad)), frame_w)
    y1 = min(int(np.ceil(ys.max() + pad)), frame_h)
    region_w, region_h = x1 - x0, y1 - y0
    if region_w <= 0 or region_h <= 0:
        return frame_bgr

    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
    )
    query_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(1, -1, 2)

    _cost, transformed = tps.applyTransformation(query_points)
    garment_coords = transformed.reshape(region_h, region_w, 2)
    map_x = garment_coords[:, :, 0]
    map_y = garment_coords[:, :, 1]

    warped_region = cv2.remap(
        garment.rgba, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    warped_rgb = warped_region[:, :, :3].astype(np.float32)
    alpha = warped_region[:, :, 3:4].astype(np.float32) / 255.0

    region = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    blended_region = warped_rgb * alpha + region * (1 - alpha)

    result = frame_bgr.copy()
    result[y0:y1, x0:x1] = blended_region.astype(np.uint8)
    return result
