"""
Skirts: a single continuous piece of fabric warped as one smooth sheet via
thin-plate spline (TPS), rather than partitioned into rigid segments the
way pants are in garment_overlay_bottom.py.

Identified purely by "segments": ["seat"] with no leg segments declared -
the same signal garment_library.py already uses to route bottoms, just
with nothing else in the list. With only the 3 waist/crotch correspondences, 
TPS mathematically reduces to a plain affine fit (3 points leave zero degrees 
of freedom for bending), so a short skirt automatically renders as a
rigid transform through this same code. A long skirt that actually reaches the 
knee/ankle, gets real bending when those points are calibrated.
"""
import cv2
import numpy as np

import garment_rig

# Same 8 points calibrate.py already knows how to collect for a bottom
# (see garment_overlay_bottom.POINT_NAMES_BOTTOM) - a skirt is calibrated
# the same way, just typically with fewer of the trailing ones actually
# clickable, depending on how long the skirt is.
POINT_NAMES_SKIRT = [
    "hip_center",
    "left_waist",
    "right_waist",
    "crotch",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# calibrate.py allows saving once these 4 (hip_center, left_waist,
# right_waist, crotch) are placed. Knee/ankle are optional to click.
CORE_POINT_COUNT = 4

# A skirt only ever declares this one segment. 
# No separate leg/seat partition.
SEGMENT_REQUIRED_POINTS = {
    "seat": ["left_waist", "right_waist", "crotch"],
}

LEFT_HIP = garment_rig.LEFT_HIP
RIGHT_HIP = garment_rig.RIGHT_HIP
LEFT_KNEE = garment_rig.LEFT_KNEE
RIGHT_KNEE = garment_rig.RIGHT_KNEE
LEFT_ANKLE = garment_rig.LEFT_ANKLE
RIGHT_ANKLE = garment_rig.RIGHT_ANKLE
MIN_VISIBILITY = garment_rig.MIN_VISIBILITY

# Same synthesis as garment_overlay_bottom.py - MediaPipe has hip
# landmarks only, no waist or crotch, so both get built from the hips
# every frame. Kept as separate constants (not imported from bottoms)
# since a skirt's real waistband/drape can sit differently than pants'
# and these may need their own tuning by eye.
WAIST_RISE_RATIO = 0.65
SEAT_WIDTH_MULTIPLIER = 1.6
CROTCH_DROP_RATIO = 1.2

# name -> landmark index, opportunistic: a skirt long enough to have these
# calibrated will use them when visible; a short skirt without them
# calibrated never asks for them at all (see warp_and_blend).
OPTIONAL_LANDMARKS = {
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}

# 0 = exact interpolation (forced through every tracked point exactly, no
# slack - a single noisy point will visibly warp the whole skirt to match
# it). Raise this if jitter turns out to be a problem once this is running
# on a live feed.
TPS_REGULARIZATION = 0.0

# How far past the tracked point cloud's bounding box to still compute and
# draw the warp, as a multiple of waist width. Keeps the per-pixel TPS
# evaluation limited to a sensible region instead of the full frame, and
# avoids TPS's tendency to extrapolate wildly far from the data it was fit on.
PADDING_RATIO = 0.6


class GarmentSkirt:
    """A background-removed skirt image plus its calibrated anchor points. No segments/partition - the whole image is one TPS-warped sheet."""

    def __init__(self, image_path: str):
        self.rgba, self.anchors, self.occluders, self.segment_names = garment_rig.load_calibration(
            image_path, POINT_NAMES_SKIRT, SEGMENT_REQUIRED_POINTS,
            occluder_hint='"occluders": ["shoes"] for anything the garment should draw over.',
            segments_hint='["seat"] - a skirt only ever declares this one segment.',
        )


def get_body_points(pose_landmarks, frame_width, frame_height):
    """
    Same hip -> waist/crotch synthesis as garment_overlay_bottom.py (see
    that file for the reasoning behind WAIST_RISE_RATIO/
    SEAT_WIDTH_MULTIPLIER/CROTCH_DROP_RATIO). Knees/ankles are
    opportunistic: included only if that specific landmark is visible.
    """
    lm = pose_landmarks

    if max(LEFT_HIP, RIGHT_HIP) >= len(lm):
        garment_rig._debug_log("pose result has too few landmarks")
        return None

    if lm[LEFT_HIP].visibility < MIN_VISIBILITY or lm[RIGHT_HIP].visibility < MIN_VISIBILITY:
        garment_rig._debug_log("hips not visible enough - face the camera / adjust lighting")
        return None

    def to_px(landmark):
        return np.array([landmark.x * frame_width, landmark.y * frame_height], dtype=np.float32)

    left_hip = to_px(lm[LEFT_HIP])
    right_hip = to_px(lm[RIGHT_HIP])
    hip_center = (left_hip + right_hip) / 2.0
    hip_width = np.linalg.norm(right_hip - left_hip)

    waist_center = hip_center - np.array([0.0, hip_width * WAIST_RISE_RATIO], dtype=np.float32)

    named_points = {
        "hip_center": hip_center,
        "crotch": waist_center + np.array([0.0, hip_width * CROTCH_DROP_RATIO], dtype=np.float32),
        "left_waist": waist_center + (left_hip - hip_center) * SEAT_WIDTH_MULTIPLIER,
        "right_waist": waist_center + (right_hip - hip_center) * SEAT_WIDTH_MULTIPLIER,
    }
    garment_rig._debug_log("tracking with real hip landmarks")

    for name, idx in OPTIONAL_LANDMARKS.items():
        if idx < len(lm) and lm[idx].visibility >= MIN_VISIBILITY:
            named_points[name] = to_px(lm[idx])

    return named_points


def warp_and_blend(frame_bgr: np.ndarray, garment: GarmentSkirt, named_dst_points: dict,
                    light_from: np.ndarray = None) -> np.ndarray:
    """
    Fit a thin-plate spline from whichever of the skirt's calibrated
    anchors currently have a matching tracked body point, warp the whole
    skirt image with it, and alpha-blend onto frame_bgr within a padded
    region around the tracked points. Returns a new frame; does not
    mutate frame_bgr. Returns frame_bgr unchanged if fewer than 3
    correspondences are available.
    """
    names = [name for name in POINT_NAMES_SKIRT if name in named_dst_points and name in garment.anchors]
    if len(names) < 3:
        return frame_bgr

    src_pts = np.float32([garment.anchors[name] for name in names]).reshape(1, -1, 2)
    dst_pts = np.float32([named_dst_points[name] for name in names]).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(names))]

    tps = cv2.createThinPlateSplineShapeTransformer(TPS_REGULARIZATION)
    # transformingShape=dst_pts (frame space), targetShape=src_pts (garment
    # space) — so applyTransformation() on frame-space query points below
    # returns the corresponding garment-space coordinates, exactly the
    # "where do I sample from" mapping cv2.remap needs.
    tps.estimateTransformation(dst_pts, src_pts, matches)

    frame_h, frame_w = frame_bgr.shape[:2]

    xs, ys = dst_pts[0, :, 0], dst_pts[0, :, 1]
    waist_width = np.linalg.norm(
        named_dst_points["right_waist"] - named_dst_points["left_waist"]
    )
    pad = max(waist_width * PADDING_RATIO, 1.0)

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
    region_alpha = warped_region[:, :, 3].astype(np.float32) / 255.0

    if light_from is not None:
        full_alpha = np.zeros((frame_h, frame_w), dtype=np.float32)
        full_alpha[y0:y1, x0:x1] = region_alpha
        gain = garment_rig.lighting_gain(light_from, full_alpha)
        if gain is not None:
            warped_rgb = warped_rgb * gain[y0:y1, x0:x1, None]

    region = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    alpha_3 = region_alpha[:, :, None]
    blended_region = warped_rgb * alpha_3 + region * (1 - alpha_3)

    result = frame_bgr.copy()
    result[y0:y1, x0:x1] = np.clip(blended_region, 0, 255).astype(np.uint8)
    return result
