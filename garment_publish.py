"""
Everything between a photo and a garment on the mirror.

Takes bytes and a category, returns the line to tell the user - so it imports
no Telegram and the whole pipeline runs from a command line.

Usage:
    python garment_publish.py photo.jpg shirt
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import auto_anchors
import bg_remove
import composite
import config
import garment_library
import normalize
from garment_types import RIG_BY_CATEGORY


def publish(png: bytes, category: str, labels=None) -> str:
    """Normalized PNG bytes onto the mirror. Returns what to reply."""
    rig = RIG_BY_CATEGORY.get(category)
    if rig is None:
        return f"no such category {category!r} - pick one of {', '.join(RIG_BY_CATEGORY)}"
    # Refuse at upload time rather than storing a garment that can never appear.
    if rig != "top":
        return f"{category} needs the bottom rig, and nothing finds bottom anchors yet"

    cut = bg_remove.cutout(png)
    sidecar = auto_anchors.top_anchors(cv2.imdecode(np.frombuffer(cut, np.uint8), cv2.IMREAD_UNCHANGED))
    if sidecar is None:
        return "couldn't find the shoulders - try a flatter photo against a plain background"

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
    return f"{category} is up, wearing {', '.join(sidecar['segments'])}"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python garment_publish.py photo.jpg [{'|'.join(RIG_BY_CATEGORY)}]")

    with open(sys.argv[1], "rb") as f:
        print(publish(normalize.to_png(f.read()), sys.argv[2]))
