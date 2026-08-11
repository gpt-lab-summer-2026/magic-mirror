"""
Tops (torso, left/right upper arm, left/right forearm): which anchor points
calibrate.py collects, which live body points drive each of the 5 rigid
segments, and get_body_points()'s shoulder/elbow/wrist-specific tracking
logic. The generic partition-into-segments and warp/composite/relight engine
lives in garment_rig.py and is imported from there.
"""
import numpy as np

import garment_rig

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

# MediaPipe Pose landmark indices used below.
LEFT_SHOULDER = garment_rig.LEFT_SHOULDER
RIGHT_SHOULDER = garment_rig.RIGHT_SHOULDER
LEFT_ELBOW = garment_rig.LEFT_ELBOW
RIGHT_ELBOW = garment_rig.RIGHT_ELBOW
LEFT_WRIST = garment_rig.LEFT_WRIST
RIGHT_WRIST = garment_rig.RIGHT_WRIST
LEFT_HIP = garment_rig.LEFT_HIP
RIGHT_HIP = garment_rig.RIGHT_HIP

MIN_VISIBILITY = garment_rig.MIN_VISIBILITY

# name -> landmark index, for the points that are optional at runtime (an
# arm can swing out of frame without blocking the overlay entirely).
OPTIONAL_LANDMARKS = {
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
}

# Fallback used when hip landmarks aren't visible
HIP_OFFSET_RATIO = 1.2

# The 5 rigid segments, which live body points each one is driven by, and
# the draw order (later entries draw on top — forearms last, so the elbow
# joint looks clean).
SEGMENT_REQUIRED_POINTS = {
    "torso":            ["left_shoulder", "right_shoulder", "hip_center"],
    "left_upper_arm":   ["left_shoulder", "left_elbow"],
    "left_forearm":     ["left_elbow", "left_wrist"],
    "right_upper_arm":  ["right_shoulder", "right_elbow"],
    "right_forearm":    ["right_elbow", "right_wrist"],
}
SEGMENT_DRAW_ORDER = ["torso", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"]

# The torso is the one segment shaped like a triangle rather than a bone
# line — see garment_rig.build_segments/_distance_to_segment_shape.
TRIANGLE_SEGMENTS = frozenset({"torso"})

# (segment_a, segment_b, shared joint anchor name) for every place two
# segments meet. Both segments get a circular overlap zone added around
# that joint point, on top of their normal nearest-line assignment.
SEGMENT_JOINTS = [
    ("torso", "left_upper_arm", "left_shoulder"),
    ("torso", "right_upper_arm", "right_shoulder"),
    ("left_upper_arm", "left_forearm", "left_elbow"),
    ("right_upper_arm", "right_forearm", "right_elbow"),
]

# Which fitting method each segment uses. Only the torso needs the
# "correspondence-exact but structurally biggest range of motion" affine
# fit — the arm segments deliberately stay similarity-only (rotation +
# uniform scale, no shear) since that's what prevents the sleeve from
# ballooning or filling in when the arm bends.
SEGMENT_TRANSFORM_KIND = {
    "torso": "affine",
}


class Garment:
    """
    A background-removed garment image, its calibrated anchor points, and
    a one-time partition of every opaque pixel into 5 rigid segments
    (torso, left/right upper arm, left/right forearm) based purely on
    which bone line (or the torso triangle) each pixel sits closest to.
    """

    def __init__(self, image_path: str):
        self.rgba, self.anchors, self.occluders, self.segment_names = garment_rig.load_calibration(
            image_path, POINT_NAMES, SEGMENT_REQUIRED_POINTS,
            occluder_hint='"occluders": ["hands", "face", "hair"] - plus "arms" if it is sleeveless.',
            segments_hint='["torso"] for a sleeveless dress.',
        )
        self.segments = garment_rig.build_segments(
            self.rgba, self.anchors, self.segment_names, SEGMENT_REQUIRED_POINTS, SEGMENT_JOINTS,
            triangle_segments=TRIANGLE_SEGMENTS,
        )


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
        garment_rig._debug_log("pose result has too few landmarks")
        return None

    if lm[LEFT_SHOULDER].visibility < MIN_VISIBILITY or lm[RIGHT_SHOULDER].visibility < MIN_VISIBILITY:
        garment_rig._debug_log("shoulders not visible enough - face the camera / adjust lighting")
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
        garment_rig._debug_log("tracking with real hip landmarks")
    else:
        shoulder_mid = (named_points["left_shoulder"] + named_points["right_shoulder"]) / 2.0
        shoulder_width = np.linalg.norm(named_points["right_shoulder"] - named_points["left_shoulder"])
        named_points["hip_center"] = shoulder_mid + np.array(
            [0.0, shoulder_width * HIP_OFFSET_RATIO], dtype=np.float32
        )
        garment_rig._debug_log("hips out of frame - using synthetic hip-center estimate")

    for name, idx in OPTIONAL_LANDMARKS.items():
        if idx < len(lm) and lm[idx].visibility >= MIN_VISIBILITY:
            named_points[name] = to_px(lm[idx])

    return named_points


def warp_and_blend(frame_bgr: np.ndarray, garment: Garment, named_dst_points: dict,
                   light_from: np.ndarray = None) -> np.ndarray:
    """
    Warp each of the garment's 5 rigid segments onto frame_bgr — see
    garment_rig.warp_and_blend for the mechanics shared with bottoms.
    """
    return garment_rig.warp_and_blend(
        frame_bgr, garment.segments, named_dst_points,
        SEGMENT_DRAW_ORDER, SEGMENT_REQUIRED_POINTS,
        segment_transform_kind=SEGMENT_TRANSFORM_KIND, light_from=light_from,
    )
