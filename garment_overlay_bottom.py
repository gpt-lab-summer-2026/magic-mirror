"""
Bottoms (left/right upper leg, left/right lower leg): which anchor points
calibrate.py collects, which live body points drive each of the 4 rigid
segments, and get_body_points()'s hip/knee/ankle-specific tracking logic.
The generic partition-into-segments and warp/composite/relight engine is 
imported from garment_rig.py.
"""
import numpy as np

import garment_rig

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

# MediaPipe Pose landmark indices used below.
LEFT_HIP = garment_rig.LEFT_HIP
RIGHT_HIP = garment_rig.RIGHT_HIP
LEFT_KNEE = garment_rig.LEFT_KNEE
RIGHT_KNEE = garment_rig.RIGHT_KNEE
LEFT_ANKLE = garment_rig.LEFT_ANKLE
RIGHT_ANKLE = garment_rig.RIGHT_ANKLE

MIN_VISIBILITY = garment_rig.MIN_VISIBILITY

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
# joint looks clean).
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
        self.rgba, self.anchors, self.occluders, self.segment_names = garment_rig.load_calibration(
            image_path, POINT_NAMES_BOTTOM, SEGMENT_REQUIRED_POINTS,
            occluder_hint='"occluders": ["shoes"] for anything the garment should draw over.',
            segments_hint='["left_upper_leg", "right_upper_leg"] for a skirt.',
        )
        self.segments = garment_rig.build_segments(
            self.rgba, self.anchors, self.segment_names, SEGMENT_REQUIRED_POINTS, SEGMENT_JOINTS,
        )


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
        garment_rig._debug_log("pose result has too few landmarks")
        return None

    if lm[LEFT_HIP].visibility < MIN_VISIBILITY or lm[RIGHT_HIP].visibility < MIN_VISIBILITY:
        garment_rig._debug_log("hips not visible enough - face the camera / adjust lighting")
        return None

    def to_px(landmark):
        return np.array([landmark.x * frame_width, landmark.y * frame_height], dtype=np.float32)

    named_points = {
        "left_hip": to_px(lm[LEFT_HIP]),
        "right_hip": to_px(lm[RIGHT_HIP]),
    }
    named_points["hip_center"] = (named_points["left_hip"] + named_points["right_hip"]) / 2.0
    garment_rig._debug_log("tracking with real hip landmarks")

    for name, idx in OPTIONAL_LANDMARKS.items():
        if idx < len(lm) and lm[idx].visibility >= MIN_VISIBILITY:
            named_points[name] = to_px(lm[idx])

    return named_points


def warp_and_blend(frame_bgr: np.ndarray, garment: GarmentBottom, named_dst_points: dict,
                   light_from: np.ndarray = None) -> np.ndarray:
    """
    Warp each of the garment's rigid leg segments onto frame_bgr — see
    garment_rig.warp_and_blend for the mechanics shared with tops.
    """
    return garment_rig.warp_and_blend(
        frame_bgr, garment.segments, named_dst_points,
        SEGMENT_DRAW_ORDER, SEGMENT_REQUIRED_POINTS, light_from=light_from,
    )
