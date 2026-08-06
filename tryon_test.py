"""Put a garment cutout on a parsed person. The first look at what this actually produces.

The compositing lives in composite.py, shared with the webcam loop. This file only picks
the inputs and shows the before/after.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from composite import apply_garment, prepare_garment
from human_parser import load_parser, parse, class_indices

PERSON_PATH = sys.argv[1] if len(sys.argv) > 1 else 'test-images/standing_pose.png'
OUTPUT_DIR = Path('test-images-output')

# The cutout, the class whose region it covers, how much wider and longer than that region it
# hangs, and the parts redrawn over it afterwards because they sit in front of the body.
# She wears jeans, not a skirt, so a skirt has to target 'pants'.
GARMENT = {
    'path': 'test-images-output/bg_pink_skirt.out.png',
    'target': 'pants',
    'width': 1.3,
    'length': 1.3,
    'occluders': ['hands', 'arms', 'face', 'hair'],
}


person = cv2.imread(PERSON_PATH, cv2.IMREAD_COLOR)
if person is None:
    raise SystemExit(f'could not read {PERSON_PATH}')

model, processor, device = load_parser()
class_map = parse(model, processor, person, device)
garment = prepare_garment(GARMENT, class_indices(model))

result = apply_garment(person, class_map, garment)
if result is None:
    raise SystemExit(f"could not place the garment on {GARMENT['target']} in {PERSON_PATH}")

comparison = np.hstack([person, result])
OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / f'{Path(PERSON_PATH).stem}_tryon.png'), comparison)

scale = min(1.0, 900 / comparison.shape[0])
cv2.imshow('before | after', cv2.resize(comparison, None, fx=scale, fy=scale))
cv2.waitKey(0)
cv2.destroyAllWindows()
