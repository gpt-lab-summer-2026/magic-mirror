"""
Model against heuristics, on the photos neither was fitted to.

Both methods produce a whole sidecar, so both are scored on the whole thing:
every anchor in reference spans, and whether the segments list came out right -
a garment with the correct points and the wrong segments still wears badly.

Only the anchors a garment's segments actually read are scored. A sleeveless
dress carries an elbow nothing draws, and averaging it in hides a bad shoulder
behind a lucky wrist.

Usage:
    python score_anchors.py [folder]     # defaults to dataset/
"""
import sys

import cv2
import torch

import anchor_predict
import auto_anchors
from anchor_model import span_errors
from garment_dataset import DATASET_DIR, GarmentPoints, crop_points, garment_crop
from garment_overlay import POINT_NAMES, SEGMENT_REQUIRED_POINTS

METHODS = {"heuristics": auto_anchors.top_anchors, "model": anchor_predict.top_anchors}


def read_by_rig(truth):
    """Which of the 7 anchors this garment's segments actually read at runtime."""
    used = {name for segment in truth.get("segments", [])
            for name in SEGMENT_REQUIRED_POINTS.get(segment, ())}
    return torch.tensor([name in (used or set(POINT_NAMES)) for name in POINT_NAMES])


def score(examples, predict):
    """Per-anchor errors, which of them count, segment agreement, and the misses."""
    errors, counts, agreed, unreadable = [], [], [], []
    for png, truth in examples:
        rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        guess = predict(rgba)
        if guess is None:
            unreadable.append(png.name)
            continue
        crop = garment_crop(rgba)
        errors.append(span_errors(torch.from_numpy(crop_points(guess, crop)).reshape(1, -1),
                                  torch.from_numpy(crop_points(truth, crop)).reshape(1, -1))[0])
        counts.append(read_by_rig(truth))
        agreed.append(guess["segments"] == truth.get("segments"))
    if not errors:
        empty = torch.zeros(0, len(POINT_NAMES))
        return empty, empty.bool(), agreed, unreadable
    return torch.stack(errors), torch.stack(counts), agreed, unreadable


def baseline(dataset):
    """What the heuristics score on a set: the one number a model has to beat."""
    errors, counts, _, _ = score(dataset.examples, auto_anchors.top_anchors)
    return float(errors[counts].mean()) if counts.any() else float("nan")


def column(errors, counts, index=None):
    values = errors[counts] if index is None else errors[:, index][counts[:, index]]
    return f"{float(values.mean()):13.3f}" if values.numel() else f"{'-':>13s}"


def main(directory):
    validation = GarmentPoints(directory, validation=True)
    if not len(validation):
        sys.exit(f"No validation garments in {directory}/ - label more with "
                 f"python label_garments.py")

    results = {name: score(validation.examples, predict) for name, predict in METHODS.items()}

    print(f"{len(validation)} validation garments in {directory}/, error in reference spans")
    print("a dash means no garment's segments read that anchor, so it never renders\n")
    print(f"{'anchor':16s}" + "".join(f"{name:>13s}" for name in METHODS))
    for i, point in enumerate(POINT_NAMES):
        print(f"{point:16s}" + "".join(column(*results[n][:2], i) for n in METHODS))
    print(f"{'':16s}" + "".join(f"{'':->13s}" for _ in METHODS))
    print(f"{'mean':16s}" + "".join(column(*results[n][:2]) for n in METHODS))

    print(f"\n{'segments right':16s}" + "".join(
        f"{sum(results[n][2])}/{len(results[n][2]):<11d}" for n in METHODS))
    for name, (_, _, _, unreadable) in results.items():
        if unreadable:
            print(f"{name} could not read {len(unreadable)}: {', '.join(unreadable[:5])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR)
