"""
Shared rigging engine behind both garment_overlay.py (tops: torso, arms) and
garment_overlay_bottom.py (bottoms: legs). Nothing in this file knows which
body region it's serving — every function takes the specific anchor names,
segment definitions, and draw order as arguments, so the two overlay modules
stay the only place that says "torso" or "left_upper_leg".

Only the per-body-region config (which points, which segments, which joints) 
differ in garment_overlay.py and garment_overlay_bottom.py.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Pose landmark indices used across both body regions.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

MIN_VISIBILITY = 0.5

# How far each segment's mask extends past its base (nearest-line)
# boundary at a shared joint, as a multiple of that segment's own typical
# half-width. >1.0 means the rounded cap is a bit larger than the limb's
# own thickness, like a paper-doll rivet, which comfortably covers
# the joint at any bend angle.
JOINT_OVERLAP_MULTIPLIER = 1.15

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


def _distance_to_segment_shape(name, pts, anchors, segment_required_points, triangle_segments):
    """Segments named in triangle_segments are the triangle between their 3
    required points (e.g. torso); every other segment is the line between
    its 2 required points (an arm or leg)."""
    required = [anchors[n] for n in segment_required_points[name]]
    if name in triangle_segments:
        a, b, c = required
        return _point_to_triangle_distance(pts, a, b, c)
    start, end = required
    return _point_to_segment_distance(pts, start, end)


def fit_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Least-squares rotation + uniform scale + translation from src_pts to
    dst_pts (Umeyama's method) — no shear, no independent x/y scaling.
    Works for exactly 2 points (limb segments, exact fit) or more
    (least-squares fit). Returns a 2x3 matrix usable with cv2.warpAffine.
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
    points — an exact solve, unlike similarity. Used for the torso, where 
    orcing one uniform scale across both the shoulder-width and shoulder-to-hip 
    directions was causing it to shrink whenever the live body's proportions
    didn't exactly match the calibration photo's.
    """
    return cv2.getAffineTransform(
        np.asarray(src_pts, dtype=np.float32),
        np.asarray(dst_pts, dtype=np.float32),
    )


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


def load_calibration(image_path: str, point_names: list, segment_required_points: dict,
                      occluder_hint: str, segments_hint: str):
    """
    Load a garment's RGBA image and its calibrated <name>.anchors.json,
    validating both against point_names/segment_required_points. Raises
    FileNotFoundError/ValueError with a message pointing at calibrate.py on
    any problem. Returns (rgba, anchors, occluders, segment_names).

    occluder_hint/segments_hint are the body-region-specific example text
    for the two calibration keys calibrate.py doesn't write itself.
    """
    rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(f"Could not read garment image: {image_path}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Garment image must have an alpha channel (RGBA): {image_path}")

    anchors_path = str(Path(image_path).with_suffix("")) + ".anchors.json"
    if not Path(anchors_path).exists():
        raise FileNotFoundError(
            f"No calibration found for {image_path}. "
            f"Run: python calibrate.py {image_path}"
        )
    with open(anchors_path) as f:
        raw_anchors = json.load(f)

    # calibrate.py does not write this key, so a newly calibrated garment has
    # to have one added by hand. Deliberate: defaulting it would let a hidden
    # body part silently show through the fabric.
    if "occluders" not in raw_anchors:
        raise ValueError(
            f"Calibration file for {image_path} has no \"occluders\" list. Add e.g. "
            f'{occluder_hint}'
        )

    # Not defaulted: a segment left on a garment that doesn't cover that body
    # part steals fabric down its side and flies it off on the wearer's limb.
    if "segments" not in raw_anchors:
        raise ValueError(
            f'Calibration file for {image_path} has no "segments" list. Add the parts '
            f'this garment actually covers, e.g. {segments_hint} '
            f'Valid names: {", ".join(segment_required_points)}'
        )
    unknown = [s for s in raw_anchors["segments"] if s not in segment_required_points]
    if unknown:
        raise ValueError(
            f"Calibration file for {image_path} lists unknown segments {unknown}. "
            f'Valid names: {", ".join(segment_required_points)}'
        )

    # Only the points actually needed by this garment's own declared
    # segments have to exist — not every point_names entry. A sleeveless
    # top or a skirt is expected to be missing the arm/leg points
    # entirely, since calibrate.py lets you save without them.
    needed_points = {n for seg in raw_anchors["segments"] for n in segment_required_points[seg]}
    missing = [name for name in needed_points if name not in raw_anchors]
    if missing:
        raise ValueError(
            f"Calibration file for {image_path} declares segments {raw_anchors['segments']} "
            f"but is missing the points they need: {missing}. "
            f"Re-run: python calibrate.py {image_path}"
        )

    anchors = {name: np.float32(raw_anchors[name]) for name in point_names if name in raw_anchors}
    return rgba, anchors, raw_anchors["occluders"], raw_anchors["segments"]


def build_segments(rgba: np.ndarray, anchors: dict, segment_names: list,
                    segment_required_points: dict, segment_joints: list,
                    triangle_segments: frozenset = frozenset()):
    """
    One-time partition of every opaque pixel in rgba into rigid segments:
    each pixel is assigned to whichever bone line (or triangle, for a name
    in triangle_segments) it's geometrically closest to, then joints get a
    rounded overlap zone added to both neighboring segments' masks. Returns
    {segment_name: {rgba, bbox, src_points} or None}.
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    distances = np.stack(
        [_distance_to_segment_shape(name, pts, anchors, segment_required_points, triangle_segments)
         for name in segment_names], axis=1)
    labels = np.argmin(distances, axis=1)

    opaque = (alpha.ravel() > 0)

    # Base assignment: each pixel belongs to whichever bone/triangle
    # it's nearest to — no overlap yet.
    seg_masks = {}
    half_width = {}
    for i, name in enumerate(segment_names):
        mask = (labels == i) & opaque
        seg_masks[name] = mask
        # A robust "typical half-width" for this segment: the 75th
        # percentile of how far its own pixels sit from its own bone, used
        # only to size that segment's joint overlap radius below.
        own_dist = distances[mask, i]
        half_width[name] = float(np.percentile(own_dist, 75)) if own_dist.size else 0.0

    # Rounded, overlapping joints: add a circular disk of pixels around each
    # shared joint point to BOTH neighboring segments' masks, on top of the
    # base assignment above. A triangle segment (e.g. torso) has no single
    # "half-width" of its own, so its joint radius borrows the other side's.
    for seg_a, seg_b, joint_name in segment_joints:
        if seg_a not in seg_masks or seg_b not in seg_masks:
            continue  # a joint only exists where both of its segments do
        joint_point = anchors[joint_name]
        if seg_a in triangle_segments:
            radius = JOINT_OVERLAP_MULTIPLIER * half_width[seg_b]
        elif seg_b in triangle_segments:
            radius = JOINT_OVERLAP_MULTIPLIER * half_width[seg_a]
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

        seg_rgba = rgba[y0:y1, x0:x1].copy()
        local_mask = seg_mask[y0:y1, x0:x1]
        seg_rgba[~local_mask, 3] = 0  # zero alpha outside this segment, even within the bbox

        required_names = segment_required_points[name]
        src_points = np.float32([anchors[n] for n in required_names])

        segments[name] = dict(rgba=seg_rgba, bbox=(x0, y0, x1, y1), src_points=src_points)
    return segments


def _composite_segment_over(canvas_premult, canvas_alpha, seg_rgba, bbox, M):
    """
    Warp seg_rgba (already cropped + masked to just this segment) with the
    rigid transform M, and alpha-composite it "over" the accumulated
    canvas at the right location, clipped to the canvas bounds.

    Returns the canvas rect it wrote to, or None if it wrote nothing.
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
        return None  # sanity guard against a degenerate transform blowing up the canvas size

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
        return None
    lx0, ly0 = cx0 - dx0, cy0 - dy0
    lx1, ly1 = cx1 - dx0, cy1 - dy0

    patch_rgb = warped[ly0:ly1, lx0:lx1, :3].astype(np.float32)
    patch_alpha = warped[ly0:ly1, lx0:lx1, 3].astype(np.float32) / 255.0

    region_premult = canvas_premult[cy0:cy1, cx0:cx1]
    region_alpha = canvas_alpha[cy0:cy1, cx0:cx1]

    a = patch_alpha[:, :, None]
    region_premult[:] = patch_rgb * a + region_premult * (1 - a)
    region_alpha[:] = patch_alpha + region_alpha * (1 - patch_alpha)
    return cx0, cy0, cx1, cy1


LIGHT_SCALE = 8        # fit at 1/8 resolution
LIGHT_GAIN_MIN = 0.7   # widen the pair for a stronger effect
LIGHT_GAIN_MAX = 1.3


def lighting_gain(frame_bgr: np.ndarray, garment_alpha: np.ndarray):
    """
    Brightness multiplier putting the room's light back on the garment, None
    if it covers nothing. A plane, not a blur - a blur would reproduce the albedo
    underneath and blow the sleeves out.
    """
    h, w = garment_alpha.shape
    sw, sh = w // LIGHT_SCALE, h // LIGHT_SCALE

    tiny = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weight = cv2.resize(garment_alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    if weight.sum() < 1.0:
        return None

    ys, xs = np.mgrid[0:sh, 0:sw]
    basis = np.stack([np.ones(sh * sw), (xs / sw * 2 - 1).ravel(), (ys / sh * 2 - 1).ravel()],
                     axis=1).astype(np.float32)
    weighted = basis * weight.reshape(-1, 1)
    coeffs, *_ = np.linalg.lstsq(weighted.T @ basis, weighted.T @ gray.ravel(), rcond=None)
    fitted = (basis @ coeffs).reshape(sh, sw)

    mean = float((fitted * weight).sum() / weight.sum())
    if mean < 1e-3:
        return None
    gain = np.clip(fitted / mean, LIGHT_GAIN_MIN, LIGHT_GAIN_MAX)
    return cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)


def warp_and_blend(frame_bgr: np.ndarray, segments: dict, named_dst_points: dict,
                    segment_draw_order: list, segment_required_points: dict,
                    segment_transform_kind: dict = None, light_from: np.ndarray = None) -> np.ndarray:
    """
    Warp each segment (whichever have all their required live points
    currently tracked) with its own rigid transform, layering them in
    segment_draw_order, then alpha-blend the result onto frame_bgr. Returns
    a new frame; does not mutate frame_bgr.

    segment_transform_kind: {name: "affine"} for segments that need the full
    affine fit (e.g. the torso); every other/missing name uses similarity.
    light_from: clean camera frame to relight against, or None to skip.
    """
    segment_transform_kind = segment_transform_kind or {}
    h, w = frame_bgr.shape[:2]
    canvas_premult = np.zeros((h, w, 3), dtype=np.float32)
    canvas_alpha = np.zeros((h, w), dtype=np.float32)
    drawn = None   # union of every segment's rect; the canvas is empty outside it

    for name in segment_draw_order:
        seg = segments.get(name)
        if seg is None:
            continue
        required_names = segment_required_points[name]
        if not all(n in named_dst_points for n in required_names):
            continue

        dst_points = np.float32([named_dst_points[n] for n in required_names])
        if segment_transform_kind.get(name) == "affine":
            M = fit_affine_transform(seg["src_points"], dst_points)
        else:
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
