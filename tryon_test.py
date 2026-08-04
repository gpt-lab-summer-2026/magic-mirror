"""Put a garment cutout on a parsed person. The first look at what this actually produces.

Deliberately crude: the garment is stretched to the bounding box of the person's existing
top, then the body parts that belong in front are drawn back over it. The point is to see
how it fails, not to make it look good.
"""

from pathlib import Path

import cv2
import numpy as np

from human_parser import load_parser, parse, class_indices

PERSON_PATH = 'test-images/standing_pose.png'
OUTPUT_DIR = Path('test-images-output')

# The garment, and the class whose region it should cover. She wears jeans, not a
# skirt, so a skirt has to target 'pants'.
GARMENT_PATH = 'test-images-output/bg_pink_skirt.out.png'
TARGET_CLASS = 'pants'

# Redrawn over the garment afterwards, because these sit in front of the body.
OCCLUDERS = ['hands', 'arms', 'face', 'hair']


def mask_bounds(mask):
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if columns.size == 0:
        raise SystemExit('empty mask - nothing to place')
    return columns[0], rows[0], columns[-1] + 1, rows[-1] + 1


def crop_to_content(garment):
    """The cutout has transparent margins, and scaling those would waste the target box."""
    left, top, right, bottom = mask_bounds(garment[:, :, 3] > 0)
    return garment[top:bottom, left:right]


def overlay(frame, garment, bounds):
    left, top, right, _ = bounds
    width = right - left
    height = round(garment.shape[0] * width / garment.shape[1])  # keep the garment's own proportions
    bottom = min(top + height, frame.shape[0])
    resized = cv2.resize(garment, (width, height))[:bottom - top]

    alpha = resized[:, :, 3:4] / 255.0
    region = frame[top:bottom, left:right]
    frame[top:bottom, left:right] = (resized[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    return frame


person = cv2.imread(PERSON_PATH, cv2.IMREAD_COLOR)
garment = cv2.imread(GARMENT_PATH, cv2.IMREAD_UNCHANGED)
if person is None:
    raise SystemExit(f'could not read {PERSON_PATH}')
if garment is None:
    raise SystemExit(f'could not read {GARMENT_PATH} - run bg_remove.py first')
if garment.shape[2] != 4:
    raise SystemExit(f'{GARMENT_PATH} has no alpha channel')

model, processor, device = load_parser()
class_map = parse(model, processor, person, device)
index = class_indices(model)

target = class_map == index[TARGET_CLASS]
result = overlay(person.copy(), crop_to_content(garment), mask_bounds(target))
for name in OCCLUDERS:
    in_front = class_map == index[name]
    result[in_front] = person[in_front]

comparison = np.hstack([person, result])
OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / 'tryon.png'), comparison)

scale = min(1.0, 900 / comparison.shape[0])
cv2.imshow('before | after', cv2.resize(comparison, None, fx=scale, fy=scale))
cv2.waitKey(0)
cv2.destroyAllWindows()
