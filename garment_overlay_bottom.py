"""
Loading a calibrated garment and overlaying it on a camera frame each loop
iteration using per-segment rigid (rotation + uniform scale, no shear)
warps: left/right upper leg, left/right lower leg.

Each segment can only rotate and rescale around its own joints — it
structurally cannot stretch or fill in, since a similarity transform has
no freedom to change shape, only orientation and size. Which garment pixels
belong to which segment is worked out automatically from the existing
calibrated points: every opaque pixel is assigned to whichever bone line
it's geometrically closest to.

Pure geometry helpers, the similarity-transform fit, the compositing/
lighting code, and a few shared constants live in garment_overlay.py and
are imported from there rather than duplicated.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from garment_overlay import (
    JOINT_OVERLAP_MULTIPLIER,
    LEFT_HIP,
    RIGHT_HIP,
    MIN_VISIBILITY,
    _debug_log,
    _point_to_segment_distance,
    _composite_segment_over,
    fit_similarity_transform,
    lighting_gain,
)

# The full set of anchor points calibrate.py collects, in the order it
# collects them in. calibrate.py imports this directly.
#
# hip_center isn't read by any segment below (there's no waistband/torso
# segment for bottoms yet) but is still collected during calibration so a
# future waist segment can use it without re-calibrating every garment.
POINT_NAMES_BOTTOM = [
    "hip_center",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MediaPipe Pose landmark indices we care about. LEFT_HIP/RIGHT_HIP are
# imported from garment_overlay since they're the same landmarks.
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# name -> landmark index, for the points that are optional at runtime (a
# leg can swing out of frame without blocking the overlay entirely).
OPTIONAL_LANDMARKS = {
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}

# The 4 rigid segments, which live body points each one is driven by, and
# the draw order (later entries draw on top — lower legs last, so the knee
# joint looks clean rather than showing the upper-leg segment's edge).
SEGMENT_REQUIRED_POINTS = {
    "left_upper_leg":   ["left_hip", "left_knee"],
    "left_lower_leg":   ["left_knee", "left_ankle"],
    "right_upper_leg":  ["right_hip", "right_knee"],
    "right_lower_leg":  ["right_knee", "right_ankle"],
}
SEGMENT_DRAW_ORDER = ["left_upper_leg", "right_upper_leg", "left_lower_leg", "right_lower_leg"]

# (segment_a, segment_b, shared joint anchor name) for every place two
# segments meet. Both segments get a circular overlap zone added around
# that joint point, on top of their normal nearest-line assignment.
SEGMENT_JOINTS = [
    ("left_upper_leg", "left_lower_leg", "left_knee"),
    ("right_upper_leg", "right_lower_leg", "right_knee"),
]


class GarmentBottom:
    """
    A background-removed bottom garment (pants/skirt) image, its
    calibrated anchor points, and a one-time partition of every opaque
    pixel into up to 4 rigid segments (left/right upper leg, left/right
    lower leg) based on which bone line each pixel sits closest to.
    """

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

        missing = [name for name in POINT_NAMES_BOTTOM if name not in raw_anchors]
        if missing:
            raise ValueError(
                f"Calibration file for {image_path} is missing {missing}. "
                f"Re-run: python calibrate.py {image_path}"
            )

        # calibrate.py does not write this key, so a newly calibrated garment
        # has to have one added by hand. Deliberate: defaulting it would let
        # e.g. footwear silently poke out from under a pair of pants.
        if "occluders" not in raw_anchors:
            raise ValueError(
                f"Calibration file for {image_path} has no \"occluders\" list. Add e.g. "
                f'"occluders": ["shoes"] for anything the garment should draw over.'
            )

        # Not defaulted: a leg segment left on a garment that doesn't cover
        # that leg (e.g. a skirt) steals fabric down its side.
        if "segments" not in raw_anchors:
            raise ValueError(
                f'Calibration file for {image_path} has no "segments" list. Add the parts '
                f'this garment actually covers, e.g. ["left_upper_leg", "right_upper_leg"] '
                f'for a skirt. Valid names: {", ".join(SEGMENT_REQUIRED_POINTS)}'
            )
        unknown = [s for s in raw_anchors["segments"] if s not in SEGMENT_REQUIRED_POINTS]
        if unknown:
            raise ValueError(
                f"Calibration file for {image_path} lists unknown segments {unknown}. "
                f'Valid names: {", ".join(SEGMENT_REQUIRED_POINTS)}'
            )

        self.anchors = {name: np.float32(raw_anchors[name]) for name in POINT_NAMES_BOTTOM}
        self.occluders = raw_anchors["occluders"]
        self.segment_names = raw_anchors["segments"]
        self.segments = self._build_segments()

    def _build_segments(self):
        h, w = self.rgba.shape[:2]
        alpha = self.rgba[:, :, 3]
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

        a = self.anchors
        segment_names = self.segment_names
        distances = np.stack(
            [_point_to_segment_distance(pts, *(a[n] for n in SEGMENT_REQUIRED_POINTS[name]))
             for name in segment_names], axis=1)
        labels = np.argmin(distances, axis=1)

        opaque = (alpha.ravel() > 0)

        # Base assignment: each pixel belongs to whichever bone it's
        # nearest to — no overlap yet.
        seg_masks = {}
        half_width = {}
        for i, name in enumerate(segment_names):
            mask = (labels == i) & opaque
            seg_masks[name] = mask
            # A robust "typical half-width" for this segment: the 75th
            # percentile of how far its own pixels sit from its own bone,
            # used only to size that segment's joint overlap radius below.
            own_dist = distances[mask, i]
            half_width[name] = float(np.percentile(own_dist, 75)) if own_dist.size else 0.0

        # Rounded, overlapping joints: add a circular disk of pixels around
        # each shared joint point to BOTH neighboring segments' masks, on
        # top of the base assignment above.
        for seg_a, seg_b, joint_name in SEGMENT_JOINTS:
            if seg_a not in seg_masks or seg_b not in seg_masks:
                continue  # a joint only exists where both of its segments do
            joint_point = a[joint_name]
            radius = JOINT_OVERLAP_MULTIPLIER * min(half_width[seg_a], half_width[seg_b])
            dist_to_joint = np.linalg.norm(pts - joint_point, axis=1)
            near_joint = (dist_to_joint <= radius) & opaque
            seg_masks[seg_a] = seg_masks[seg_a] | near_joint
            seg_masks[seg_b] = seg_masks[seg_b] | near_joint

        segments = {}
        for name in segment_names:
            seg_mask = seg_masks[name].reshape(h, w)
            ys_idx, xs_idx = np.where(seg_mask)
            if len(xs_idx) == 0:
                segments[name] = None  # nothing assigned to this segment — skip it at runtime
                continue
            x0, y0 = int(xs_idx.min()), int(ys_idx.min())
            x1, y1 = int(xs_idx.max()) + 1, int(ys_idx.max()) + 1

            seg_rgba = self.rgba[y0:y1, x0:x1].copy()
            local_mask = seg_mask[y0:y1, x0:x1]
            seg_rgba[~local_mask, 3] = 0  # zero alpha outside this segment, even within the bbox

            required_names = SEGMENT_REQUIRED_POINTS[name]
            src_points = np.float32([self.anchors[n] for n in required_names])

            segments[name] = dict(rgba=seg_rgba, bbox=(x0, y0, x1, y1), src_points=src_points)
        return segments


def get_body_points(pose_landmarks, frame_width, frame_height):
    """
    Build a dict of currently-tracked named body points, in pixel
    coordinates. Hips are required (returns None if either is missing or
    low-confidence). Hip-center always comes back, computed from the hips.
    Knees and ankles are opportunistic: each is included only if that
    specific landmark is visible this frame.
    """
    lm = pose_landmarks

    if max(LEFT_HIP, RIGHT_HIP) >= len(lm):
        _debug_log("pose result has too few landmarks")
        return None

    if lm[LEFT_HIP].visibility < MIN_VISIBILITY or lm[RIGHT_HIP].visibility < MIN_VISIBILITY:
        _debug_log("hips not visible enough - face the camera / adjust lighting")
        return None

    def to_px(landmark):
        return np.array([landmark.x * frame_width, landmark.y * frame_height], dtype=np.float32)

    named_points = {
        "left_hip": to_px(lm[LEFT_HIP]),
        "right_hip": to_px(lm[RIGHT_HIP]),
    }
    named_points["hip_center"] = (named_points["left_hip"] + named_points["right_hip"]) / 2.0
    _debug_log("tracking with real hip landmarks")

    for name, idx in OPTIONAL_LANDMARKS.items():
        if idx < len(lm) and lm[idx].visibility >= MIN_VISIBILITY:
            named_points[name] = to_px(lm[idx])

    return named_points


def warp_and_blend(frame_bgr: np.ndarray, garment: GarmentBottom, named_dst_points: dict,
                   light_from: np.ndarray = None) -> np.ndarray:
    """
    Warp each of the garment's rigid leg segments (whichever have all their
    required live points currently tracked) with its own rotation+scale
    transform, layering them in SEGMENT_DRAW_ORDER, then alpha-blend the
    result onto frame_bgr. Returns a new frame; does not mutate frame_bgr.

    light_from: clean camera frame to relight against, or None to skip.
    """
    h, w = frame_bgr.shape[:2]
    canvas_premult = np.zeros((h, w, 3), dtype=np.float32)
    canvas_alpha = np.zeros((h, w), dtype=np.float32)
    drawn = None   # union of every segment's rect; the canvas is empty outside it

    for name in SEGMENT_DRAW_ORDER:
        seg = garment.segments.get(name)
        if seg is None:
            continue
        required_names = SEGMENT_REQUIRED_POINTS[name]
        if not all(n in named_dst_points for n in required_names):
            continue

        dst_points = np.float32([named_dst_points[n] for n in required_names])
        M = fit_similarity_transform(seg["src_points"], dst_points)
        rect = _composite_segment_over(canvas_premult, canvas_alpha, seg["rgba"], seg["bbox"], M)
        if rect is not None:
            drawn = rect if drawn is None else (
                min(drawn[0], rect[0]), min(drawn[1], rect[1]),
                max(drawn[2], rect[2]), max(drawn[3], rect[3]))

    if light_from is not None and drawn is not None:
        gain = lighting_gain(light_from, canvas_alpha)
        if gain is not None:
            x0, y0, x1, y1 = drawn
            canvas_premult[y0:y1, x0:x1] *= gain[y0:y1, x0:x1, None]

    alpha = canvas_alpha[:, :, None]
    blended = canvas_premult + frame_bgr.astype(np.float32) * (1 - alpha)
    # Clip: a gain above 1 can push premultiplied RGB past 255, which wraps.
    return np.clip(blended, 0, 255).astype(np.uint8)
