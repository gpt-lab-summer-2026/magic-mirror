"""
Every garment the demo can show.

A list and an index. load() scans config.GARMENT_DIR once at startup and
builds every Garment up front - one small RGBA and its segment partition
each - because a visible pause when the user presses an arrow is what
makes a demo feel broken.
"""
import json
from pathlib import Path

import config
import garment_overlay
import garment_overlay_bottom

GARMENTS = []
_index = 0

# Top and bottom segment names never overlap (torso/arms vs. seat/legs), so
# whether a garment's own declared "segments" list intersects this set is
# an unambiguous, filename-independent way to tell which overlay module it
# needs - unlike guessing from the filename, which has no entry for
# shorts, dresses named oddly, or anything typed in a different case.
_BOTTOM_SEGMENT_NAMES = frozenset(garment_overlay_bottom.SEGMENT_REQUIRED_POINTS)


def _is_bottom(anchors_path: Path) -> bool:
    with open(anchors_path) as f:
        declared = set(json.load(f).get("segments", []))
    return bool(declared & _BOTTOM_SEGMENT_NAMES)


def load(directory: str = config.GARMENT_DIR):
    """
    Every .png with a matching .anchors.json, in filename order. Which
    overlay module a garment needs is read from its own "segments" list
    (see _is_bottom), not guessed from the filename. A garment that fails
    to load - bad or incomplete calibration - is skipped with a message
    rather than taking every other garment down with it; one broken file
    shouldn't turn an arrow-key press into a crash.
    """
    for png in sorted(Path(directory).glob("*.png")):
        anchors_path = png.with_suffix(".anchors.json")
        if not anchors_path.exists():
            print(f"Skipping {png.name}: no {anchors_path.name} - run python calibrate.py {png}", flush=True)
            continue
        try:
            garment = (garment_overlay_bottom.GarmentBottom(str(png)) if _is_bottom(anchors_path)
                       else garment_overlay.Garment(str(png)))
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {png.name}: {e}", flush=True)
            continue
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
