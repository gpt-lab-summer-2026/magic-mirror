"""
What has to be true of a sidecar before it reaches the mirror.

Not an accuracy test - nothing here knows where the shoulders really are. These
are the shapes a garment cannot be: shoulders out of level, a span of nothing,
hips above the collar. They exist because two different things now produce
sidecars, and choosing between them means being able to recognise a broken one.

Elbows and wrists are deliberately not required to sit on fabric. A short
sleeve's elbow is past the cuff and a sleeveless garment's wrist hangs in open
air, and both of those are correct.
"""
import numpy as np

from auto_anchors import ALPHA_CUTOFF

MAX_TILT = 0.15            # of shoulder width. Clicked garments: 0.003 to 0.035
SPAN_RANGE = (0.25, 1.2)   # of the garment's full width. Clicked: 0.51 to 0.68
DROP_RANGE = (0.3, 3.0)    # hips below the shoulder line, in spans. Clicked: 0.71 to 1.93
ON_FABRIC = ("left_shoulder", "right_shoulder", "hip_center")


def problems(sidecar, rgba):
    """Every reason this sidecar should not be published. Empty means it may be."""
    mask = rgba[:, :, 3] > ALPHA_CUTOFF
    columns = np.flatnonzero(mask.any(0))
    if not columns.size:
        return ["there is no garment in the cutout"]

    left, right, hip = (sidecar[name] for name in ON_FABRIC)
    found = [f"left_{joint} is not left of right_{joint}"
             for joint in ("shoulder", "elbow", "wrist")
             if sidecar[f"left_{joint}"][0] >= sidecar[f"right_{joint}"][0]]

    span = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    if span < 1:
        return found + ["both shoulders landed in the same place"]

    share = span / (columns[-1] - columns[0])
    if not SPAN_RANGE[0] <= share <= SPAN_RANGE[1]:
        found.append(f"the shoulders span {share:.2f} of the garment's width")

    tilt = abs(left[1] - right[1]) / span
    if tilt > MAX_TILT:
        found.append(f"the shoulders are {tilt:.2f} spans out of level")

    drop = (hip[1] - (left[1] + right[1]) / 2) / span
    if not DROP_RANGE[0] <= drop <= DROP_RANGE[1]:
        found.append(f"hip_center sits {drop:.2f} spans below the shoulders")

    height, width = mask.shape
    for name in ON_FABRIC:
        x, y = sidecar[name]
        if not (0 <= x < width and 0 <= y < height and mask[y, x]):
            found.append(f"{name} is not on the garment")
    return found
