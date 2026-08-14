"""
Everything between a photo and a garment on the mirror.

Takes bytes and a category, returns the line to tell the user - so it imports
no Telegram and the whole pipeline runs from a command line.

Usage:
    python garment_publish.py photo.jpg shirt
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import anchor_check
import anchor_predict
import auto_anchors
import bg_remove
import composite
import config
import garment_library
import normalize
from garment_dataset import DATASET_DIR
from garment_overlay import POINT_NAMES
from garment_types import RIG_BY_CATEGORY

# Everything garment_library needs to build a garment. A sidecar missing any of
# it is half-written, and must not reach the mirror.
SIDECAR_KEYS = POINT_NAMES + ["segments", "occluders"]

# The trained model first, the silhouette second. Both have to pass the same
# check, so a checkpoint that has learned nothing cannot quietly take over.
ANCHOR_SOURCES = (anchor_predict.top_anchors, auto_anchors.top_anchors)


def training_copy(cut: bytes) -> Path:
    """Where this cutout lives in the training set: named by its own contents, so
    the same photo twice is one example - and so a correction made once is found again."""
    return Path(DATASET_DIR) / f"{hashlib.md5(cut).hexdigest()[:12]}.png"


def corrected_anchors(cut: bytes):
    """The sidecar you fixed by hand for this exact photo, if you fixed one."""
    sidecar = training_copy(cut).with_suffix(".anchors.json")
    if not sidecar.exists():
        return None
    with open(sidecar) as f:
        saved = json.load(f)
    # Half-written or hand-edited by mistake: fall through to measuring rather
    # than taking the mirror down over a file the bot did not write.
    return saved if all(key in saved for key in SIDECAR_KEYS) else None


def anchors_for(rgba, corrected=None):
    """The first sidecar that passes the check - your correction, then the model,
    then the silhouette. Returns (sidecar, complaint) with exactly one of them set,
    so a garment is never published unmeasured."""
    complaint = "couldn't find the shoulders - try a flatter photo against a plain background"
    if corrected is not None and not anchor_check.problems(corrected, rgba):
        return corrected, None
    for source in ANCHOR_SOURCES:
        sidecar = source(rgba)
        if sidecar is None:
            continue
        found = anchor_check.problems(sidecar, rgba)
        if not found:
            return sidecar, None
        complaint = f"that photo came out wrong: {found[0]}"
    return None, complaint


def keep_for_training(cut: bytes):
    """Every published cutout, kept to be labelled later. Only the image is
    written, never the sidecar beside it - that one is yours."""
    png = training_copy(cut)
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(cut)


def publish(png: bytes, category: str, labels=None) -> str:
    """Normalized PNG bytes onto the mirror. Returns what to reply."""
    rig = RIG_BY_CATEGORY.get(category)
    if rig is None:
        return f"no such category {category!r} - pick one of {', '.join(RIG_BY_CATEGORY)}"
    # Refuse at upload time rather than storing a garment that can never appear.
    if rig != "top":
        return f"{category} needs the bottom rig, and nothing finds bottom anchors yet"

    cut = bg_remove.cutout(png)
    corrected = corrected_anchors(cut)
    sidecar, complaint = anchors_for(
        cv2.imdecode(np.frombuffer(cut, np.uint8), cv2.IMREAD_UNCHANGED), corrected)
    if sidecar is None:
        return complaint
    keep_for_training(cut)

    # Garment reads an image path and derives its sidecar path from it, so both
    # have to exist on disk, adjacent, sharing a stem. The category is the name,
    # which is also why no string a user typed ever reaches a path.
    directory = Path(config.BOT_GARMENT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{category}.png"
    image_path.write_bytes(cut)
    with open(image_path.with_suffix(".anchors.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    garment = garment_library.build(image_path)
    if labels is not None:
        # Before publishing, never after: the render loop reads the LUT with no guard.
        try:
            garment.occluder_lut = composite.resolve_occluders(garment, labels)
        except ValueError as e:
            return str(e)
    garment_library.replace_or_add(garment)
    measured = "your saved anchors" if sidecar is corrected else "auto anchors"
    return f"{category} is up on {measured}, wearing {', '.join(sidecar['segments'])}"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python garment_publish.py photo.jpg [{'|'.join(RIG_BY_CATEGORY)}]")

    with open(sys.argv[1], "rb") as f:
        print(publish(normalize.to_png(f.read()), sys.argv[2]))
