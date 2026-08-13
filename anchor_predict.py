"""
The trained model, wearing auto_anchors' interface.

top_anchors(rgba) returns the same complete sidecar auto_anchors.top_anchors
does, so garment_publish can pick between them without knowing which is which,
and fall back when there is no checkpoint to load.

The 7 points come from the model. Segments and occluders are still measured, but
from the garment mask along the arms the model has just placed - a much steadier
reading than finding the sleeve in the first place, which is the part the
silhouette could not do.
"""
from pathlib import Path

import numpy as np
import torch

from anchor_model import CHECKPOINT, AnchorNet
from auto_anchors import ALPHA_CUTOFF, BASE_OCCLUDERS, LONG_SLEEVE
from garment_dataset import crop_image, garment_crop, to_tensor
from garment_overlay import POINT_NAMES

# Of shoulder width. Fabric reaching less far than this down the arm is an
# armhole, not a sleeve. Measured on the clicked garments: sleeveless 0.01-0.05,
# a short sleeve 0.42, long sleeves 1.03-1.35.
MIN_SLEEVE = 0.25
SAMPLES = 100   # points tested along each arm
GAP = 4         # consecutive empty samples that end a sleeve; fewer is a buttonhole

_model = None


def load(checkpoint=CHECKPOINT):
    """The trained model, loaded once. None while there is no checkpoint to load."""
    global _model
    if _model is None and Path(checkpoint).exists():
        _model = AnchorNet()
        _model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        _model.eval()
    return _model


def predict(rgba, model):
    """The 7 anchors in source pixels, straight off the model."""
    crop = garment_crop(rgba)
    with torch.inference_mode():
        predicted = model(to_tensor(crop_image(rgba, crop)).unsqueeze(0))
    x0, y0, side = crop
    return {name: (x0 + x * side, y0 + y * side)
            for name, (x, y) in zip(POINT_NAMES, predicted[0].reshape(-1, 2).tolist())}


def fabric_reach(mask, shoulder, wrist):
    """How far the garment reaches from a shoulder towards its wrist, as a
    fraction of the way there. Read along the arm rather than across the
    silhouette, so it measures the sleeve where the wearer's arm will be."""
    height, width = mask.shape
    xs = np.linspace(shoulder[0], wrist[0], SAMPLES).round().astype(int).clip(0, width - 1)
    ys = np.linspace(shoulder[1], wrist[1], SAMPLES).round().astype(int).clip(0, height - 1)
    empty = 0
    for i, solid in enumerate(mask[ys, xs]):
        empty = 0 if solid else empty + 1
        if empty >= GAP:
            return (i - GAP + 1) / (SAMPLES - 1)
    return 1.0


def sleeve_reach(mask, points, side, width):
    """One sleeve's length in shoulder widths, the currency auto_anchors thinks in."""
    shoulder, wrist = points[f"{side}_shoulder"], points[f"{side}_wrist"]
    arm = np.hypot(wrist[0] - shoulder[0], wrist[1] - shoulder[1])
    return fabric_reach(mask, shoulder, wrist) * arm / width


def top_anchors(rgba):
    """The complete sidecar in source pixels, or None when there is no model yet."""
    model = load()
    if model is None:
        return None

    points = predict(rgba, model)
    # Guarded: a model that collapsed every anchor onto one spot must not divide by nothing.
    width = max(abs(points["right_shoulder"][0] - points["left_shoulder"][0]), 1.0)
    mask = rgba[:, :, 3] > ALPHA_CUTOFF

    segments = ["torso"]
    for side in ("left", "right"):
        reach = sleeve_reach(mask, points, side, width)
        if reach < MIN_SLEEVE:
            continue
        segments.append(f"{side}_upper_arm")
        # Only where the sleeve really gets there, or the rig carries cuff down a bare arm.
        if reach >= LONG_SLEEVE:
            segments.append(f"{side}_forearm")

    sidecar = {name: [int(x), int(y)] for name, (x, y) in points.items()}
    sidecar["rig"] = "top"
    sidecar["segments"] = segments
    sidecar["occluders"] = BASE_OCCLUDERS if len(segments) > 1 else BASE_OCCLUDERS + ["arms"]
    return sidecar
