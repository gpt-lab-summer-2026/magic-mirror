"""
Every garment the demo can show.

A list and an index. load() scans config.GARMENT_DIR once at startup and
builds every Garment up front - one small RGBA and its segment partition
each.
"""
import json
from pathlib import Path

import config
import garment_overlay
import garment_overlay_bottom
import garment_overlay_skirt

GARMENTS = []
_index = 0

# Top and bottom segment names never overlap (torso/arms vs. seat/legs), so
# whether a garment's own declared "segments" list intersects this set is
# an unambiguous, filename-independent way to tell which overlay module it
# needs.
_BOTTOM_SEGMENT_NAMES = frozenset(garment_overlay_bottom.SEGMENT_REQUIRED_POINTS)


def _garment_class(anchors_path: Path):
    """
    Which class a garment needs, read purely from its own declared "segments"

    "segments": ["seat"] and NOTHING else = GarmentSkirt. This has to
    be checked before the general bottom check below, since "seat" is
    also one of GarmentBottom's segment names. This is also correct for
    a seat-only garment that isn't conceptually a skirt (e.g. very short 
    shorts): GarmentSkirt's thin-plate-spline warp mathematically reduces to a
    plain affine fit with only the 3 waist/crotch correspondences, so the 
    render comes out identical to the old rigid GarmentBottom path either way.

    Anything else that intersects the bottom segment names (i.e. declares
    at least one leg segment alongside "seat") is a GarmentBottom.
    Anything that doesn't touch bottom names at all is a top/dress.
    """
    with open(anchors_path) as f:
        declared = set(json.load(f).get("segments", []))
    if declared == {"seat"}:
        return garment_overlay_skirt.GarmentSkirt
    if declared & _BOTTOM_SEGMENT_NAMES:
        return garment_overlay_bottom.GarmentBottom
    return garment_overlay.Garment


def load(directory: str = config.GARMENT_DIR):
    """
    Every .png with a matching .anchors.json, in filename order. Which
    overlay module a garment needs is read from its own "segments" list
    (see _garment_class). A garment that fails to load - bad or incomplete 
    calibration - is skipped with a message.
    """
    for png in sorted(Path(directory).glob("*.png")):
        anchors_path = png.with_suffix(".anchors.json")
        if not anchors_path.exists():
            print(f"Skipping {png.name}: no {anchors_path.name} - run python calibrate.py {png}", flush=True)
            continue
        try:
            garment_class = _garment_class(anchors_path)
            garment = garment_class(str(png))
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
