"""Repaint body parts that sit in front of a placed garment.

Placement comes from the affine_transform branch; this module copies the
parser's occluder classes from the camera frame back on top of it.
"""

import numpy as np

# A sleeved garment must not list 'arms', or its sleeves get painted over.
OCCLUDERS = {
    'test-images-output/w_black_dress.out.png': ['hands', 'arms', 'face', 'hair'],
    'test-images-output/w_pink_skirt.out.png': ['hands', 'arms', 'face', 'hair'],
    'test-images-output/w_white_top.out.png': ['hands', 'face', 'hair'],
}


def apply_occluders(placed, frame, class_map, occluder_indices):
    """frame must be the clean camera frame, not an annotated one."""
    in_front = np.isin(class_map, occluder_indices)
    placed[in_front] = frame[in_front]
    return placed
