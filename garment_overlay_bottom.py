"""
Bottoms (seat, left/right upper leg, left/right lower leg): which anchor
points calibrate.py collects, which live body points drive each of the 5
rigid segments, and get_body_points()'s hip/knee/ankle-specific tracking
logic. The generic partition-into-segments and warp/composite/relight engine
is imported from garment_rig.py.

The waist-to-crotch area is one triangular "seat" segment - like the tops
file's torso, it needs an affine fit so waist width and waist-to-crotch
depth can scale independently. The legs only become separate rigid pieces
below the crotch, which is where fabric actually splits on a real garment.

MediaPipe tracks hip joints, not a waist or a crotch, so get_body_points()
synthesizes both every frame - see WAIST_RISE_RATIO, SEAT_WIDTH_MULTIPLIER,
and CROTCH_DROP_RATIO for the tunable knobs.
"""
import numpy as np

import garment_rig
from garment_rig import _point_to_segment_distance, _point_to_triangle_distance, JOINT_OVERLAP_MULTIPLIER

# The full set of anchor points calibrate.py collects, in the order it
# collects them in. calibrate.py imports this directly.
#
# left_waist/right_waist are the waistband's top corners on the flat-lay
# garment photo - there's no hip joint to click on a photo, so this was
# always what got clicked here even back when the key was named "hip".
# hip_center isn't read by any segment below, but is still collected during
# calibration in case a future waistband shape wants it.
POINT_NAMES_BOTTOM = [
    "hip_center",
    "left_waist",
    "right_waist",
    "crotch",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# calibrate.py allows saving once this many points are placed, in order —
# everything after this is optional to click, matching everything the seat
# panel itself needs (hip_center, left_waist, right_waist, crotch). The leg
# points beyond that are exactly as optional to calibrate as they already
# are to track live: a skirt's "segments" list just won't include the leg
# segments, so nothing downstream ever needs their anchors.
CORE_POINT_COUNT = 4

# MediaPipe Pose landmark indices used below.
LEFT_HIP = garment_rig.LEFT_HIP
RIGHT_HIP = garment_rig.RIGHT_HIP
LEFT_KNEE = garment_rig.LEFT_KNEE
RIGHT_KNEE = garment_rig.RIGHT_KNEE
LEFT_ANKLE = garment_rig.LEFT_ANKLE
RIGHT_ANKLE = garment_rig.RIGHT_ANKLE

MIN_VISIBILITY = garment_rig.MIN_VISIBILITY

# --- MediaPipe has hip landmarks only - no waist, no crotch - so both get
# synthesized every frame from the hips, sized relative to hip width
# (always available, unlike knees which can be out of frame). All three
# ratios are rough estimates: tune them by eye against the live feed. ---

# The hip joints sit visibly below where a real waistband actually rests,
# which is what makes an unraised fit look like it's sliding down the hips.
# Raises left_waist/right_waist upward from the hip line before anything
# else is measured from them.
WAIST_RISE_RATIO = 0.5

# MediaPipe's hip landmarks are also visibly narrower than where a real
# pair of pants drapes - widen the two waist points the seat panel is fit
# against, outward from their own midpoint, so the waist doesn't render
# too narrow.
SEAT_WIDTH_MULTIPLIER = 1.55

# Crotch depth below the (already-raised) waist line.
CROTCH_DROP_RATIO = 0.9

# name -> landmark index, for the points that are optional at runtime (a
# leg can swing out of frame without blocking the overlay entirely).
OPTIONAL_LANDMARKS = {
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}

# The 5 rigid segments, which live body points each one is driven by, and
# the draw order (later entries draw on top — lower legs last, so the knee
# joint looks clean rather than showing the upper-leg segment's edge).
SEGMENT_REQUIRED_POINTS = {
    "seat":             ["left_waist", "right_waist", "crotch"],
    "left_upper_leg":   ["crotch", "left_knee"],
    "left_lower_leg":   ["left_knee", "left_ankle"],
    "right_upper_leg":  ["crotch", "right_knee"],
    "right_lower_leg":  ["right_knee", "right_ankle"],
}
SEGMENT_DRAW_ORDER = ["left_upper_leg", "right_upper_leg", "seat", "left_lower_leg", "right_lower_leg"]

# The seat is the one segment shaped like a triangle rather than a bone
# line — see garment_rig.build_segments/_distance_to_segment_shape.
TRIANGLE_SEGMENTS = frozenset({"seat"})

# (segment_a, segment_b, shared joint anchor name) for every place two
# segments meet. Both segments get a circular overlap zone added around
# that joint point, on top of their normal nearest-line assignment.
SEGMENT_JOINTS = [
    ("seat", "left_upper_leg", "crotch"),
    ("seat", "right_upper_leg", "crotch"),
    ("left_upper_leg", "left_lower_leg", "left_knee"),
    ("right_upper_leg", "right_lower_leg", "right_knee"),
]

# Which fitting method each segment uses. Only the seat needs the
# "correspondence-exact but structurally biggest range of motion" affine
# fit — the leg segments deliberately stay similarity-only (rotation +
# uniform scale, no shear), which is what keeps the pant leg from
# ballooning or filling in when the knee bends.
SEGMENT_TRANSFORM_KIND = {
    "seat": "affine",
}


# The crotch joint (leg meeting the triangular seat) needs to be WIDE
# (perpendicular to the leg's own bone line) to cover a baggy thigh's full
# width without pinching - that's what CROTCH_JOINT_PERCENTILE (measuring
# a wider percentile than the standard half_width) is for. But a plain
# circle sized that wide is equally tall in every direction, including
# straight up the bone toward the waist - which reaches unnaturally far
# into the seat's own territory. An ellipse decouples the two: wide
# sideways (CROTCH_JOINT_PERCENTILE), limited along the bone (the
# standard half_width, same value already used for the knee joint).
CROTCH_JOINT_PERCENTILE = 95


def _elliptical_joint_mask(pts, joint_point, bone_unit, radius_along, radius_perp):
    """1.0 exactly on the ellipse boundary; <1.0 inside."""
    perp_unit = np.array([-bone_unit[1], bone_unit[0]])
    offset = pts - joint_point
    along = offset @ bone_unit
    perp = offset @ perp_unit
    return (along / max(radius_along, 1e-6)) ** 2 + (perp / max(radius_perp, 1e-6)) ** 2


def _build_bottom_segments(rgba, anchors, segment_names, segment_required_points, segment_joints):
    """
    Bottoms-only variant of garment_rig.build_segments: identical Voronoi
    partition and joint-overlap-disk mechanics, except the crotch joint
    (leg meeting the triangular seat) uses an ellipse instead of a circle -
    see CROTCH_JOINT_PERCENTILE / _elliptical_joint_mask. The knee joint is
    untouched: still a plain circle, same formula as always. Deliberately
    not in garment_rig.py, so tops/torso/arms are completely unaffected.
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    opaque = (alpha.ravel() > 0)

    def raw_distance(name):
        required = [anchors[n] for n in segment_required_points[name]]
        if name in TRIANGLE_SEGMENTS:
            a, b, c = required
            return _point_to_triangle_distance(pts, a, b, c)
        start, end = required
        return _point_to_segment_distance(pts, start, end)

    distances = np.stack([raw_distance(name) for name in segment_names], axis=1)
    labels = np.argmin(distances, axis=1)

    seg_masks = {}
    half_width = {}
    half_width_at_triangle_joint = {}
    for i, name in enumerate(segment_names):
        mask = (labels == i) & opaque
        seg_masks[name] = mask
        own_dist = distances[mask, i]
        half_width[name] = float(np.percentile(own_dist, 75)) if own_dist.size else 0.0
        half_width_at_triangle_joint[name] = (
            float(np.percentile(own_dist, CROTCH_JOINT_PERCENTILE)) if own_dist.size else 0.0
        )

    for seg_a, seg_b, joint_name in segment_joints:
        if seg_a not in seg_masks or seg_b not in seg_masks:
            continue
        joint_point = anchors[joint_name]
        if seg_a in TRIANGLE_SEGMENTS or seg_b in TRIANGLE_SEGMENTS:
            leg_name = seg_b if seg_a in TRIANGLE_SEGMENTS else seg_a
            leg_start, leg_end = [anchors[n] for n in segment_required_points[leg_name]]
            bone = leg_end - leg_start
            bone_unit = bone / np.linalg.norm(bone)
            radius_along = JOINT_OVERLAP_MULTIPLIER * half_width[leg_name]
            radius_perp = JOINT_OVERLAP_MULTIPLIER * half_width_at_triangle_joint[leg_name]
            ell = _elliptical_joint_mask(pts, joint_point, bone_unit, radius_along, radius_perp)
            near_joint = (ell <= 1.0) & opaque
        else:
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
            segments[name] = None
            continue
        x0, y0 = int(xs_idx.min()), int(ys_idx.min())
        x1, y1 = int(xs_idx.max()) + 1, int(ys_idx.max()) + 1
        seg_rgba = rgba[y0:y1, x0:x1].copy()
        local_mask = seg_mask[y0:y1, x0:x1]
        seg_rgba[~local_mask, 3] = 0
        required_names = segment_required_points[name]
        src_points = np.float32([anchors[n] for n in required_names])
        segments[name] = dict(rgba=seg_rgba, bbox=(x0, y0, x1, y1), src_points=src_points)
    return segments


class GarmentBottom:
    """
    A background-removed bottom garment (pants/skirt) image, its
    calibrated anchor points, and a one-time partition of every opaque
    pixel into up to 5 rigid segments (seat, left/right upper leg,
    left/right lower leg) based on which bone line (or the seat triangle)
    each pixel sits closest to.
    """

    def __init__(self, image_path: str):
        self.rgba, self.anchors, self.occluders, self.segment_names = garment_rig.load_calibration(
            image_path, POINT_NAMES_BOTTOM, SEGMENT_REQUIRED_POINTS,
            occluder_hint='"occluders": ["shoes"] for anything the garment should draw over.',
            segments_hint='["seat", "left_upper_leg", "right_upper_leg"] for shorts.',
        )
        self.segments = _build_bottom_segments(
            self.rgba, self.anchors, self.segment_names, SEGMENT_REQUIRED_POINTS, SEGMENT_JOINTS,
        )


def get_body_points(pose_landmarks, frame_width, frame_height):
    """
    Build a dict of currently-tracked named body points, in pixel
    coordinates. Hips are required (returns None if either is missing or
    low-confidence) but never returned directly - MediaPipe has no waist or
    crotch landmark, so left_waist/right_waist/crotch are all synthesized
    from the hips (see WAIST_RISE_RATIO/SEAT_WIDTH_MULTIPLIER/
    CROTCH_DROP_RATIO). hip_center is the hips' own real midpoint, kept for
    reference. Knees and ankles are opportunistic: each is included only if
    that specific landmark is visible this frame.
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

    # Raise the waist line above the hip joints first - everything else
    # (crotch depth, waist width) is then measured from that raised line,
    # not the raw hip line, so the whole seat panel moves up together.
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


def warp_and_blend(frame_bgr: np.ndarray, garment: GarmentBottom, named_dst_points: dict,
                   light_from: np.ndarray = None) -> np.ndarray:
    """
    Warp each of the garment's rigid segments onto frame_bgr — see
    garment_rig.warp_and_blend for the mechanics shared with tops.
    """
    return garment_rig.warp_and_blend(
        frame_bgr, garment.segments, named_dst_points,
        SEGMENT_DRAW_ORDER, SEGMENT_REQUIRED_POINTS,
        segment_transform_kind=SEGMENT_TRANSFORM_KIND, light_from=light_from,
    )
