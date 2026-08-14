"""Repaint body parts that sit in front of a placed garment."""

import cv2
import numpy as np

# this currently handles tops, I'd add only hands for bottoms?
def resolve_occluders(garment, labels):
    """Occluder names from .anchors.json -> a 256-entry LUT, 255 where that class
    sits in front. Raises on a typo: an unmatched name silently disables occlusion."""
    # Raise rather than exit: a bad name reaching this from the bot thread mid-demo
    # must not take the mirror down, and SystemExit skips every except Exception.
    unknown = [name for name in garment.occluders if name not in labels]
    if unknown:
        raise ValueError(f"{garment.name}.anchors.json lists unknown occluders {unknown}. "
                         f"Valid names: {', '.join(sorted(labels))}")
    lut = np.zeros(256, dtype=np.uint8)
    lut[[labels[name] for name in garment.occluders]] = 255
    return lut


def apply_occluders(placed, frame, class_map, occluder_lut):
    """frame must be the clean camera frame, not an annotated one."""
    # copyTo, not np.isin: 3.5 ms vs 32 at 1280x960, same output.
    return cv2.copyTo(frame, occluder_lut[class_map], placed)
