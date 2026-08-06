"""Measure FASHN Human Parser latency and per-class coverage on still images.

Pass image files or a folder to compare framings; with no arguments it falls back to the
studio shot. Run it on the GPU box and on a laptop to get both numbers — the GPU number
decides whether masks can be refreshed every few frames or only once per person.

SegFormer-B4, 18 classes, 384x576 input. First run downloads ~244 MB from HuggingFace.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from human_parser import load_parser, parse

DEFAULT_PATH = 'test-images/standing_pose.png'
OUTPUT_DIR = Path('test-images-output')
RUNS = 1


def image_paths(arguments):
    """PowerShell does not expand *.png for a native command, so a folder has to expand itself."""
    if not arguments:
        return [Path(DEFAULT_PATH)]
    paths = []
    for argument in arguments:
        path = Path(argument)
        paths.extend(sorted(path.glob('*.png')) if path.is_dir() else [path])
    return paths


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


def show(image):
    """Shrunk to fit the screen; judge edge quality from the full-size copy on disk instead."""
    scale = min(1.0, 900 / image.shape[0])
    cv2.imshow('classes | blended', cv2.resize(image, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


paths = image_paths(sys.argv[1:])
if not paths:
    raise SystemExit('no .png files found')
for path in paths:
    if not path.is_file():  # checked here because loading the model first costs 20 seconds
        raise SystemExit(f'not a file: {path}')

print(f'{len(paths)} images, loading parser...')
model, processor, device = load_parser()
print(f'device:  {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

palette = build_palette(len(model.config.id2label))
OUTPUT_DIR.mkdir(exist_ok=True)

for index, path in enumerate(paths):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f'could not read {path}')

    if index == 0:
        report_latency(model, processor, frame, device)  # the same for any input, so measure once

    print(f'\n{path.name}  ({frame.shape[1]}x{frame.shape[0]})')
    class_map = parse(model, processor, frame, device)
    print_coverage(class_map, model.config.id2label)

    colored = palette[class_map]
    blended = cv2.addWeighted(frame, 0.5, colored, 0.5, 0)
    side_by_side = np.hstack([colored, blended])
    cv2.imwrite(str(OUTPUT_DIR / f'{path.stem}_classes.png'), side_by_side)

    if len(paths) == 1:
        show(side_by_side)

print(f'\nclass maps written to {OUTPUT_DIR}')
