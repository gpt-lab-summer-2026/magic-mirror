"""Check whether MediaPipe multiclass selfie segmentation holds up at full-body distance.

The model is trained on selfies, so the open question is how clean the clothes and
body-skin boundaries are on a standing full-body shot. Look at the arm edges.

Model download:
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
"""

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'selfie_multiclass_256x256.tflite'
IMAGE_PATH = 'test-images/standing_pose.png'
OUTPUT_DIR = Path('test-images-output')
WEBCAM_SIZE = (640, 480)

# List position is the class id the model outputs. Colours are BGR for OpenCV.
CLASSES = [
    ('background', (0, 0, 0)),
    ('hair', (255, 0, 255)),
    ('body-skin', (255, 255, 0)),
    ('face-skin', (0, 255, 255)),
    ('clothes', (0, 255, 0)),
    ('others', (0, 0, 255)),
]
COLORS = np.array([color for _, color in CLASSES], dtype=np.uint8)


def print_coverage(category_mask):
    for index, (name, _) in enumerate(CLASSES):
        share = np.count_nonzero(category_mask == index) / category_mask.size
        print(f'{name:>11}: {share:6.1%}')


def to_mp_image(frame):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def report_latency(segmenter, frame, label, runs=20):
    mp_image = to_mp_image(frame)
    segmenter.segment(mp_image)  # first call builds the graph, don't time it
    start = time.perf_counter()
    for _ in range(runs):
        segmenter.segment(mp_image)
    ms = (time.perf_counter() - start) / runs * 1000
    print(f'{label:>11}: {ms:6.1f} ms  ({frame.shape[1]}x{frame.shape[0]})')


frame = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if frame is None:
    raise SystemExit(f'could not read {IMAGE_PATH}')

options = vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    output_category_mask=True)

with vision.ImageSegmenter.create_from_options(options) as segmenter:
    report_latency(segmenter, frame, 'full size')
    report_latency(segmenter, cv2.resize(frame, WEBCAM_SIZE), 'webcam size')

    # numpy_view() points into the result's own memory, so copy before the result is freed.
    result = segmenter.segment(to_mp_image(frame))
    category_mask = np.copy(result.category_mask.numpy_view())

print_coverage(category_mask)

colored = COLORS[category_mask]
blended = cv2.addWeighted(frame, 0.5, colored, 0.5, 0)
side_by_side = np.hstack([colored, blended])

# Full resolution goes to disk because the on-screen copy is too shrunk to judge edges.
OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / 'segment_classes.png'), side_by_side)

scale = min(1.0, 900 / side_by_side.shape[0])
cv2.imshow('classes | blended', cv2.resize(side_by_side, None, fx=scale, fy=scale))
cv2.waitKey(0)
cv2.destroyAllWindows()
