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
import sys

import anchor_session
import bg_remove
import composite
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
    # The upload never touches the disk: it is built straight from the two
    # values it already is. The category is the name, which is also why no
    # string a user typed ever reaches a path.
    garment = garment_library.build_from_memory(
        anchor_session.decode(cutout), sidecar, name=category)
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
