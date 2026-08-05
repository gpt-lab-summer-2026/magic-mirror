"""
Loading a calibrated garment and overlaying it on a camera frame each loop
iteration using an affine transform driven by MediaPipe pose landmarks.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Pose landmark indices we care about.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

MIN_VISIBILITY = 0.5

# Fallback used when hip landmarks aren't visible — very common, since most
# webcam mirror setups frame from around the chest or waist up and hips
# never make it into the shot. Vertical distance from the shoulder midpoint
# down to the hip midpoint, expressed as a multiple of shoulder width.
# ~1.1-1.3 is a reasonable average adult proportion; nudge this if the
# synthetic point looks too high/low for your camera distance.
HIP_OFFSET_RATIO = 1.2

_last_debug_message = None


def _debug_log(message: str) -> None:
    """Print only when the tracking state changes, so this doesn't spam the console every frame."""
    global _last_debug_message
    if message != _last_debug_message:
        print(f"[garment_overlay] {message}", flush=True)
        _last_debug_message = message


class Garment:
    """A background-removed garment image plus its 3 calibrated anchor points."""

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
            anchors = json.load(f)

        # Order matters: must match the order get_body_triangle() builds points in.
        self.src_points = np.float32([
            anchors["left_shoulder"],
            anchors["right_shoulder"],
            anchors["hip_center"],
        ])


class LandmarkSmoother:
    """Exponential moving average over a set of (x, y) points, to reduce jitter."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._smoothed = None

    def update(self, points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        if self._smoothed is None:
            self._smoothed = points
        else:
            self._smoothed = self.alpha * points + (1 - self.alpha) * self._smoothed
        return self._smoothed

    def reset(self):
        self._smoothed = None


def get_body_triangle(pose_landmarks, frame_width, frame_height):
    """
    Pull left shoulder, right shoulder, and hip-center from one detected
    person's landmarks, converted to pixel coordinates.

    Returns None if any required landmark is missing or below the visibility
    threshold, so the caller can fall back to the last good transform instead
    of warping onto garbage points.
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

    left_shoulder = to_px(lm[LEFT_SHOULDER])
    right_shoulder = to_px(lm[RIGHT_SHOULDER])

    hips_visible = (
        max(LEFT_HIP, RIGHT_HIP) < len(lm)
        and lm[LEFT_HIP].visibility >= MIN_VISIBILITY
        and lm[RIGHT_HIP].visibility >= MIN_VISIBILITY
    )

    if hips_visible:
        hip_center = (to_px(lm[LEFT_HIP]) + to_px(lm[RIGHT_HIP])) / 2.0
        _debug_log("tracking with real hip landmarks")
    else:
        # Hips out of frame or low-confidence: estimate a hip point straight
        # down from the shoulder midpoint instead of requiring hips at all.
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_center = shoulder_mid + np.array([0.0, shoulder_width * HIP_OFFSET_RATIO], dtype=np.float32)
        _debug_log("hips out of frame - using synthetic hip-center estimate")

    return np.float32([left_shoulder, right_shoulder, hip_center])


def warp_and_blend(frame_bgr: np.ndarray, garment: Garment, dst_points: np.ndarray) -> np.ndarray:
    """
    Warp the garment onto the body triangle and alpha-blend it onto frame_bgr.
    Returns a new frame; does not mutate frame_bgr in place.
    """
    h, w = frame_bgr.shape[:2]

    M = cv2.getAffineTransform(garment.src_points, dst_points.astype(np.float32))

    warped = cv2.warpAffine(
        garment.rgba, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    warped_rgb = warped[:, :, :3].astype(np.float32)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0

    blended = warped_rgb * alpha + frame_bgr.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)