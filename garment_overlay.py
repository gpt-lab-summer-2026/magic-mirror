"""
Loading a calibrated garment and overlaying it on a camera frame each loop
iteration using per-segment rigid (rotation + uniform scale, no shear)
warps: torso, left/right upper arm, left/right forearm.

Unlike the earlier TPS approach, each segment can only rotate and rescale
around its own joints — it structurally cannot stretch or fill in, since a
similarity transform has no freedom to change shape, only orientation and
size. Which garment pixels belong to which segment is worked out
automatically from the existing 7 calibrated points (no new points to
click): every opaque pixel is assigned to whichever bone line (or the
torso triangle) it's geometrically closest to.
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
# never make it into the shot.
HIP_OFFSET_RATIO = 1.2

# The 5 rigid segments, which live body points each one is driven by, and
# the draw order (later entries draw on top — forearms last, so the elbow
# joint looks clean rather than showing the upper-arm segment's edge).
SEGMENT_REQUIRED_POINTS = {
    "torso":            ["left_shoulder", "right_shoulder", "hip_center"],
    "left_upper_arm":   ["left_shoulder", "left_elbow"],
    "left_forearm":     ["left_elbow", "left_wrist"],
    "right_upper_arm":  ["right_shoulder", "right_elbow"],
    "right_forearm":    ["right_elbow", "right_wrist"],
}
SEGMENT_DRAW_ORDER = ["torso", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"]

# How far each segment's mask extends past its base (nearest-line)
# boundary at a shared joint, as a multiple of that segment's own typical
# half-width. >1.0 means the rounded cap is a bit larger than the limb's
# own thickness — like a real paper-doll rivet — which comfortably covers
# the joint at any bend angle rather than just barely reaching it.
JOINT_OVERLAP_MULTIPLIER = 1.15

# (segment_a, segment_b, shared joint anchor name) for every place two
# segments meet. Both segments get a circular overlap zone added around
# that joint point, on top of their normal nearest-line assignment.
SEGMENT_JOINTS = [
    ("torso", "left_upper_arm", "left_shoulder"),
    ("torso", "right_upper_arm", "right_shoulder"),
    ("left_upper_arm", "left_forearm", "left_elbow"),
    ("right_upper_arm", "right_forearm", "right_elbow"),
]

_last_debug_message = None


def _debug_log(message: str) -> None:
    """Print only when the tracking state changes, so this doesn't spam the console every frame."""
    global _last_debug_message
    if message != _last_debug_message:
        print(f"[garment_overlay] {message}", flush=True)
        _last_debug_message = message


# --- geometry helpers, used once at garment-load time to assign each pixel
# to its nearest segment, and every frame to fit each segment's rigid
# transform. ---

def _sign(p, q, r):
    return (p[:, 0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[:, 1] - r[1])


def _point_in_triangle(pts, a, b, c):
    d1, d2, d3 = _sign(pts, a, b), _sign(pts, b, c), _sign(pts, c, a)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _point_to_segment_distance(pts, a, b):
    seg = b - a
    seg_len2 = np.dot(seg, seg)
    if seg_len2 < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ seg) / seg_len2, 0.0, 1.0)
    proj = a + t[:, None] * seg
    return np.linalg.norm(pts - proj, axis=1)


def _point_to_triangle_distance(pts, a, b, c):
    inside = _point_in_triangle(pts, a, b, c)
    d_ab = _point_to_segment_distance(pts, a, b)
    d_bc = _point_to_segment_distance(pts, b, c)
    d_ca = _point_to_segment_distance(pts, c, a)
    edge_dist = np.minimum(np.minimum(d_ab, d_bc), d_ca)
    return np.where(inside, 0.0, edge_dist)


def fit_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Least-squares rotation + uniform scale + translation from src_pts to
    dst_pts (Umeyama's method) — no shear, no independent x/y scaling.
    Works for exactly 2 points (arm segments, exact fit) or more (torso,
    least-squares fit). Returns a 2x3 matrix usable with cv2.warpAffine.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n = src_pts.shape[0]
    mu_src, mu_dst = src_pts.mean(axis=0), dst_pts.mean(axis=0)
    src_c, dst_c = src_pts - mu_src, dst_pts - mu_dst
    var_src = (src_c ** 2).sum() / n
    if var_src < 1e-9:
        # Degenerate calibration (src points coincide) — fall back to identity-ish.
        return np.array([[1, 0, mu_dst[0] - mu_src[0]], [0, 1, mu_dst[1] - mu_src[1]]], dtype=np.float32)
    Sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(Sigma) < 0 or (np.linalg.det(U) * np.linalg.det(Vt) < 0):
        S[-1, -1] = -1
    R = U @ S @ Vt
    c = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - c * (R @ mu_src)
    return np.hstack([c * R, t.reshape(2, 1)]).astype(np.float32)


def fit_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Full affine (independent x/y scale + shear allowed) from exactly 3
    points — an exact solve, unlike similarity, which can only satisfy 3
    arbitrary points exactly if they happen to form geometrically similar
    triangles. Used for the torso, where forcing one uniform scale across
    both the shoulder-width and shoulder-to-hip directions was causing it
    to shrink to a "compromise" size whenever the live body's proportions
    didn't exactly match the calibration photo's.
    """
    return cv2.getAffineTransform(
        np.asarray(src_pts, dtype=np.float32),
        np.asarray(dst_pts, dtype=np.float32),
    )


# Which fitting method each segment uses. Only the torso needs the
# "correspondence-exact but structurally biggest range of motion" affine
# fit — the arm segments deliberately stay similarity-only (rotation +
# uniform scale, no shear) since that's what prevents the sleeve from
# ballooning or filling in when the arm bends.
SEGMENT_TRANSFORM_KIND = {
    "torso": "affine",
    "left_upper_arm": "similarity",
    "left_forearm": "similarity",
    "right_upper_arm": "similarity",
    "right_forearm": "similarity",
}


class Garment:
    """
    A background-removed garment image, its calibrated anchor points, and
    a one-time partition of every opaque pixel into 5 rigid segments
    (torso, left/right upper arm, left/right forearm) based purely on
    which bone line (or the torso triangle) each pixel sits closest to.
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

        missing = [name for name in POINT_NAMES if name not in raw_anchors]
        if missing:
            raise ValueError(
                f"Calibration file for {image_path} is missing {missing}. "
                f"Re-run: python calibrate.py {image_path}"
            )

        self.anchors = {name: np.float32(raw_anchors[name]) for name in POINT_NAMES}
        self.segments = self._build_segments()

    def _build_segments(self):
        h, w = self.rgba.shape[:2]
        alpha = self.rgba[:, :, 3]
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

        a = self.anchors
        segment_names = list(SEGMENT_REQUIRED_POINTS.keys())
        distances = np.stack([
            _point_to_triangle_distance(pts, a["left_shoulder"], a["right_shoulder"], a["hip_center"]),
            _point_to_segment_distance(pts, a["left_shoulder"], a["left_elbow"]),
            _point_to_segment_distance(pts, a["left_elbow"], a["left_wrist"]),
            _point_to_segment_distance(pts, a["right_shoulder"], a["right_elbow"]),
            _point_to_segment_distance(pts, a["right_elbow"], a["right_wrist"]),
        ], axis=1)
        labels = np.argmin(distances, axis=1)

        opaque = (alpha.ravel() > 0)

        # Base assignment: each pixel belongs to whichever bone (or the
        # torso triangle) it's nearest to — no overlap yet.
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
            joint_point = a[joint_name]
            if seg_a == "torso":
                radius = JOINT_OVERLAP_MULTIPLIER * half_width[seg_b]
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


class LandmarkSmoother:
    """
    Exponential moving average over a named set of points, to reduce
    jitter. Handles the point set changing size/composition frame to frame
    (e.g. a wrist appearing or disappearing) by smoothing each name
    independently against its own history.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._smoothed = {}

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


def _composite_segment_over(canvas_premult, canvas_alpha, seg_rgba, bbox, M):
    """
    Warp seg_rgba (already cropped + masked to just this segment) with the
    rigid transform M, and alpha-composite it "over" the accumulated
    canvas at the right location, clipped to the canvas bounds.
    """
    frame_h, frame_w = canvas_alpha.shape[:2]
    x0, y0, x1, y1 = bbox

    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    transformed = (M[:, :2] @ corners.T).T + M[:, 2]

    dx0 = int(np.floor(transformed[:, 0].min()))
    dy0 = int(np.floor(transformed[:, 1].min()))
    dx1 = int(np.ceil(transformed[:, 0].max()))
    dy1 = int(np.ceil(transformed[:, 1].max()))
    dw, dh = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
    if dw > frame_w * 4 or dh > frame_h * 4:
        return  # sanity guard against a degenerate transform blowing up the canvas size

    # Shift M so it maps seg_rgba's own local (0,0)-origin coords directly
    # into the destination bbox's local coords.
    new_translation = M[:, :2] @ np.array([x0, y0], dtype=np.float32) + M[:, 2] - np.array([dx0, dy0], dtype=np.float32)
    M_local = np.hstack([M[:, :2], new_translation.reshape(2, 1)]).astype(np.float32)

    warped = cv2.warpAffine(
        seg_rgba, M_local, (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    cx0, cy0 = max(dx0, 0), max(dy0, 0)
    cx1, cy1 = min(dx0 + dw, frame_w), min(dy0 + dh, frame_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    lx0, ly0 = cx0 - dx0, cy0 - dy0
    lx1, ly1 = cx1 - dx0, cy1 - dy0

    patch_rgb = warped[ly0:ly1, lx0:lx1, :3].astype(np.float32)
    patch_alpha = warped[ly0:ly1, lx0:lx1, 3].astype(np.float32) / 255.0

    region_premult = canvas_premult[cy0:cy1, cx0:cx1]
    region_alpha = canvas_alpha[cy0:cy1, cx0:cx1]

    a = patch_alpha[:, :, None]
    region_premult[:] = patch_rgb * a + region_premult * (1 - a)
    region_alpha[:] = patch_alpha + region_alpha * (1 - patch_alpha)


def warp_and_blend(frame_bgr: np.ndarray, garment: Garment, named_dst_points: dict) -> np.ndarray:
    """
    Warp each of the garment's 5 rigid segments (whichever have all their
    required live points currently tracked) with its own rotation+scale
    transform, layering them in SEGMENT_DRAW_ORDER, then alpha-blend the
    result onto frame_bgr. Returns a new frame; does not mutate frame_bgr.
    """
    h, w = frame_bgr.shape[:2]
    canvas_premult = np.zeros((h, w, 3), dtype=np.float32)
    canvas_alpha = np.zeros((h, w), dtype=np.float32)

    for name in SEGMENT_DRAW_ORDER:
        seg = garment.segments.get(name)
        if seg is None:
            continue
        required_names = SEGMENT_REQUIRED_POINTS[name]
        if not all(n in named_dst_points for n in required_names):
            continue

        dst_points = np.float32([named_dst_points[n] for n in required_names])
        kind = SEGMENT_TRANSFORM_KIND[name]
        if kind == "affine":
            M = fit_affine_transform(seg["src_points"], dst_points)
        else:
            M = fit_similarity_transform(seg["src_points"], dst_points)
        _composite_segment_over(canvas_premult, canvas_alpha, seg["rgba"], seg["bbox"], M)

    alpha = canvas_alpha[:, :, None]
    blended = canvas_premult + frame_bgr.astype(np.float32) * (1 - alpha)
    return blended.astype(np.uint8)
