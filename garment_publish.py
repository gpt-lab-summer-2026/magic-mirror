"""
Everything between a photo and a garment on the mirror.

Takes bytes and a category, returns the line to tell the user - so it imports
no Telegram and the whole pipeline runs from a command line.

Split in two because the anchor page happens in the middle: prepare() produces
the cutout and a draft to drag around, finish() takes whatever anchors came
back. publish() is both of them, for the command line and for a draft the user
accepted as it stands.

Usage:
    python garment_publish.py photo.jpg shirt
"""
import json
import sys
from pathlib import Path

import anchor_session
import bg_remove
import composite
import config
import garment_library
import normalize
from garment_types import RIG_BY_CATEGORY


def prepare(png: bytes, category: str):
    """Photo bytes to a cutout, its draft anchors, and whether those were measured."""
    cutout = bg_remove.cutout(png)
    sidecar, trusted = anchor_session.initial_sidecar(anchor_session.decode(cutout), category)
    return cutout, sidecar, trusted


def finish(cutout: bytes, sidecar: dict, category: str, labels=None) -> str:
    """A cutout and the anchors it ended up with, onto the mirror. Returns what to reply."""
    # Garment reads an image path and derives its sidecar path from it, so both
    # have to exist on disk, adjacent, sharing a stem. The category is the name,
    # which is also why no string a user typed ever reaches a path.
    directory = Path(config.BOT_GARMENT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{category}.png"
    image_path.write_bytes(cutout)
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


def publish(png: bytes, category: str, labels=None) -> str:
    """Normalized PNG bytes onto the mirror, with the anchors the draft came with."""
    cutout, sidecar, _ = prepare(png, category)
    return finish(cutout, sidecar, category, labels)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python garment_publish.py photo.jpg [{'|'.join(RIG_BY_CATEGORY)}]")

    with open(sys.argv[1], "rb") as f:
        try:
            print(publish(normalize.to_png(f.read()), sys.argv[2]))
        except ValueError as e:
            sys.exit(str(e))   # an unknown category or an empty cutout, in one line
