"""
Labelled cutouts in, tensors a keypoint model can train on out.

Each garment is cropped to its own silhouette and resized to one square, so the
model never has to cope with a garment being photographed near or far, left or
right of frame. Anchors travel with the crop and come out as fractions of it,
which is what makes a 900px phone photo and a 1600px product shot the same
problem.

Usage:
    python garment_dataset.py [folder]    # summary + a contact sheet to look at
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from garment_overlay import POINT_NAMES

DATASET_DIR = "dataset"
IMAGE_SIZE = 256      # what the backbone sees
CROP_MARGIN = 0.08    # of the garment's own box, so a cuff on the edge keeps some air around it
VAL_SHARE = 0.2
BACKGROUND = 128      # flat grey behind the cutout: constant, so the silhouette stays readable

ROTATION = 10.0       # degrees. A garment photo is always upright, so this stays modest
SCALE_JITTER = 0.12
SHIFT_JITTER = 0.06   # of the crop
BRIGHTNESS = 0.2


def mirrored_name(name):
    """left_wrist <-> right_wrist: flipping a photo swaps which side each anchor belongs to."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


# Flipping the pixels is not enough - without this the model learns to put the
# left wrist wherever the right one was, which trains out to the garment's middle.
MIRROR_ORDER = [POINT_NAMES.index(mirrored_name(name)) for name in POINT_NAMES]


def load_examples(directory):
    """Every cutout in `directory` that has been labelled with all 7 anchors."""
    examples = []
    for png in sorted(Path(directory).glob("*.png")):
        sidecar = png.with_suffix(".anchors.json")
        if not sidecar.exists():
            continue
        with open(sidecar) as f:
            anchors = json.load(f)
        if all(name in anchors for name in POINT_NAMES):
            examples.append((png, anchors))
    return examples


def is_validation(png):
    """A stable split: a photo lands in the same half every run, and adding
    photos never reshuffles the ones already scored."""
    digest = hashlib.md5(png.name.encode()).hexdigest()
    return int(digest, 16) % 100 < VAL_SHARE * 100


def garment_crop(rgba):
    """The square the model sees, as (x0, y0, side). Read off the alpha alone,
    because at inference time there are no anchors to crop around."""
    ys, xs = np.nonzero(rgba[:, :, 3] > 127)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(x1 - x0, y1 - y0) * (1 + 2 * CROP_MARGIN)
    return (x0 + x1 - side) / 2, (y0 + y1 - side) / 2, side


def crop_image(rgba, crop):
    """The crop, composited on flat grey, at IMAGE_SIZE. BGR, as OpenCV reads it."""
    x0, y0, side = crop
    zoom = IMAGE_SIZE / side
    M = np.float32([[zoom, 0, -x0 * zoom], [0, zoom, -y0 * zoom]])
    square = cv2.warpAffine(rgba, M, (IMAGE_SIZE, IMAGE_SIZE),
                            flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
    alpha = square[:, :, 3:4].astype(np.float32) / 255
    return (square[:, :, :3] * alpha + BACKGROUND * (1 - alpha)).astype(np.uint8)


def crop_points(anchors, crop):
    """The 7 anchors as fractions of the crop. Outside 0..1 is legal - a short
    sleeve's wrist is off the garment, which is exactly where it should be."""
    x0, y0, side = crop
    return np.float32([[(anchors[name][0] - x0) / side, (anchors[name][1] - y0) / side]
                       for name in POINT_NAMES])


def to_tensor(image):
    """A crop as the model eats it. Here rather than in the Dataset, so training
    and inference cannot come to disagree about channel order."""
    rgb = np.ascontiguousarray(image[:, :, ::-1])   # cv2 reads BGR, backbones expect RGB
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255


def augment(image, points, rng):
    """One garment, photographed slightly differently."""
    if rng.random() < 0.5:
        image = image[:, ::-1]
        points = np.stack([1 - points[:, 0], points[:, 1]], axis=1)[MIRROR_ORDER]

    M = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2),
                                rng.uniform(-ROTATION, ROTATION),
                                1 + rng.uniform(-SCALE_JITTER, SCALE_JITTER))
    M[:, 2] += rng.uniform(-SHIFT_JITTER, SHIFT_JITTER, 2) * IMAGE_SIZE
    image = cv2.warpAffine(np.ascontiguousarray(image), M, (IMAGE_SIZE, IMAGE_SIZE),
                           borderValue=(BACKGROUND,) * 3)
    points = (points * IMAGE_SIZE @ M[:, :2].T + M[:, 2]) / IMAGE_SIZE

    gain = 1 + rng.uniform(-BRIGHTNESS, BRIGHTNESS)
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8), points


class GarmentPoints(Dataset):
    """Labelled cutouts as (3 x 256 x 256 image, 14 coordinates)."""

    def __init__(self, directory=DATASET_DIR, validation=False):
        self.examples = [e for e in load_examples(directory)
                         if is_validation(e[0]) == validation]
        self.jitter = not validation   # the validation half is scored on the photo as taken

    def __len__(self):
        return len(self.examples)

    def sample(self, i, rng=None):
        """The image and points as pictures, before they become tensors - so the
        contact sheet and __getitem__ can never drift apart."""
        png, anchors = self.examples[i]
        rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        crop = garment_crop(rgba)
        image, points = crop_image(rgba, crop), crop_points(anchors, crop)
        if self.jitter:
            image, points = augment(image, points, rng or np.random.default_rng())
        return image, points

    def __getitem__(self, i):
        image, points = self.sample(i)
        return to_tensor(image), torch.from_numpy(points.reshape(-1).copy()).float()


def contact_sheet(dataset, rng, columns=4):
    """Samples drawn with their anchors, so a swapped left and right is visible."""
    tiles = []
    for i in range(min(len(dataset), 2 * columns)):
        image, points = dataset.sample(i, rng)
        tile = image.copy()
        for (x, y), name in zip(points * IMAGE_SIZE, POINT_NAMES):
            colour = (80, 80, 255) if name.startswith("left") else (80, 255, 80)
            cv2.circle(tile, (int(x), int(y)), 4, colour, -1)
        tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30)))
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 30))
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR
    train, validation = GarmentPoints(directory), GarmentPoints(directory, validation=True)
    print(f"{directory}/: {len(train)} to train on, {len(validation)} to score against")
    if not len(train):
        sys.exit("Nothing labelled yet - run python label_garments.py first.")

    out = Path("ignore") / "dataset_contact_sheet.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), contact_sheet(train, np.random.default_rng(0)))
    print(f"wrote {out} - red is left_*, green is right_*")
