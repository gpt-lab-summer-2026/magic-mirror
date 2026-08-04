"""Measure FASHN Human Parser on one still image, on whatever device is available.

Run it on the GPU box and on a laptop to get both numbers. The GPU number decides
whether masks can be refreshed every few frames or only once per person.

SegFormer-B4, 18 classes, 384x576 input. First run downloads ~244 MB from HuggingFace.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from human_parser import load_parser, parse

IMAGE_PATH = 'test-images/standing_pose.png'
OUTPUT_DIR = Path('test-images-output')
RUNS = 5


def build_palette(count):
    """One evenly spaced hue per class, background forced to black."""
    hues = np.linspace(0, 179, count, endpoint=False).astype(np.uint8)
    full = np.full(count, 255, np.uint8)
    palette = cv2.cvtColor(np.stack([hues, full, full], axis=1)[None], cv2.COLOR_HSV2BGR)[0]
    palette[0] = 0
    return palette


def report_latency(model, processor, frame, device):
    parse(model, processor, frame, device)  # warm up: first call allocates and builds kernels
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(RUNS):
        parse(model, processor, frame, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()  # kernels are async, so time only after they finish
    print(f'latency: {(time.perf_counter() - start) / RUNS * 1000:6.1f} ms')


def print_coverage(class_map, labels):
    for index, name in sorted(labels.items()):
        share = np.count_nonzero(class_map == index) / class_map.size
        print(f'{name:>11}: {share:6.1%}')


frame = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if frame is None:
    raise SystemExit(f'could not read {IMAGE_PATH}')

model, processor, device = load_parser()

print(f'device:  {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')
report_latency(model, processor, frame, device)

class_map = parse(model, processor, frame, device)
print_coverage(class_map, model.config.id2label)

colored = build_palette(len(model.config.id2label))[class_map]
blended = cv2.addWeighted(frame, 0.5, colored, 0.5, 0)
side_by_side = np.hstack([colored, blended])

OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / 'parser_classes.png'), side_by_side)

scale = min(1.0, 900 / side_by_side.shape[0])
cv2.imshow('classes | blended', cv2.resize(side_by_side, None, fx=scale, fy=scale))
cv2.waitKey(0)
cv2.destroyAllWindows()
