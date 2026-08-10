"""
Every garment the demo can show.

A list and an index. load() scans config.GARMENT_DIR once at startup and
builds every Garment up front - one small RGBA and its segment partition
each - because a visible pause when the user presses an arrow is what
makes a demo feel broken.
"""
from pathlib import Path

import config
import garment_overlay

GARMENTS = []
_index = 0


def load(directory: str = config.GARMENT_DIR):
    """Every .png with a matching .anchors.json, in filename order."""
    for png in sorted(Path(directory).glob("*.png")):
        anchors = png.with_suffix(".anchors.json")
        if not anchors.exists():
            print(f"Skipping {png.name}: no {anchors.name} - run python calibrate.py {png}", flush=True)
            continue
        garment = garment_overlay.Garment(str(png))
        # The filename stem is the on-screen label. Garment itself has no
        # use for a name, so it is attached here instead of in the class.
        garment.name = png.stem
        GARMENTS.append(garment)
    return GARMENTS


def current():
    """None when nothing loadable was found - the loop then skips the overlay."""
    return GARMENTS[_index] if GARMENTS else None


def next():
    global _index
    if GARMENTS:
        _index = (_index + 1) % len(GARMENTS)


def previous():
    global _index
    if GARMENTS:
        _index = (_index - 1) % len(GARMENTS)


def select(i: int):
    """An index with no garment behind it does nothing, so 5 on a 3-garment folder is a no-op."""
    global _index
    if 0 <= i < len(GARMENTS):
        _index = i
