"""Place a garment cutout on a parsed person.

The garment is warped onto the rotated box around the class it should cover, so position,
scale and tilt all come from the mask itself - no pose landmarks. The body parts that
belong in front are then drawn back over it. Shared by the still-image test and the
webcam loop so that both composite the same way.
"""

import cv2
import numpy as np


def mask_bounds(mask):
    """None when the class is absent, because a person can step out of frame mid-loop."""
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if columns.size == 0:
        return None
    return columns[0], rows[0], columns[-1] + 1, rows[-1] + 1


def crop_to_content(garment):
    """The cutout has transparent margins, and scaling those would waste the target box."""
    bounds = mask_bounds(garment[:, :, 3] > 0)
    if bounds is None:
        raise SystemExit('garment is fully transparent')
    left, top, right, bottom = bounds
    return garment[top:bottom, left:right]


def load_garment(path):
    """Cropped once here: the transparent margins never change, so the loop should not redo it."""
    garment = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if garment is None:
        raise SystemExit(f'could not read {path} - run bg_remove.py first')
    if garment.shape[2] != 4:
        raise SystemExit(f'{path} has no alpha channel')
    return crop_to_content(garment)


def prepare_garment(record, class_index):
    """Load a garment's cutout and turn its class names into indices.

    Done once at startup, so that neither the still test nor the loop repeats the lookup and
    the rest of this file never needs to know what the classes are called.
    """
    return {
        'image': load_garment(record['path']),
        'target': class_index[record['target']],
        'occluders': [class_index[name] for name in record['occluders']],
        'background': class_index['background'],  # carried so the pose check needs no lookup
        'width': record['width'],
        'length': record['length'],
    }


def hang_line(mask):
    """The top side of the mask's rotated box, the unit vector down the body, and its extent.

    A garment hangs from that side, so these carry position, width, tilt and length together.
    The body is assumed to run along the box's longer side, which holds for a torso or a pair
    of legs seen head-on and breaks on a region wider than it is tall.
    """
    corners = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask.astype(np.uint8))))

    first, second = corners[1] - corners[0], corners[3] - corners[0]
    longer = np.linalg.norm(first) > np.linalg.norm(second)
    along, across = (first, second) if longer else (second, first)

    down = along / np.linalg.norm(along)
    if down[1] < 0:
        down = -down  # boxPoints gives no consistent winding, so force it toward the floor

    top = corners[np.argsort(corners @ down)[:2]]
    left, right = top[np.argsort(top @ (across / np.linalg.norm(across)))]
    return left, right, down, np.linalg.norm(along)


def overlay(frame, garment, mask, width_factor, length_factor):
    left, right, down, span = hang_line(mask)

    # Both factors say how much bigger than the covered region the garment hangs. Length is
    # measured off the body, not off the cutout, whose proportions depend on how a garment
    # happened to be laid out for its photograph.
    length = span * length_factor
    centre, half = (left + right) / 2, (right - left) / 2 * width_factor
    left, right = centre - half, centre + half

    source = np.float32([[0, 0], [garment.shape[1], 0], [0, garment.shape[0]]])
    destination = np.float32([left, right, left + down * length])
    warped = cv2.warpAffine(garment, cv2.getAffineTransform(source, destination),
                            (frame.shape[1], frame.shape[0]))

    alpha = warped[:, :, 3:4] / 255.0
    frame[:] = (warped[:, :, :3] * alpha + frame * (1 - alpha)).astype(np.uint8)
    return frame


def apply_garment(frame, class_map, garment):
    """None when the garment cannot be placed, so the caller can fall back to the raw frame.

    That covers an empty frame, a person standing at an angle the warp cannot represent, and
    the covered class not being visible. Showing the plain frame beats compositing something
    visibly wrong.
    """
    bounds = mask_bounds(class_map != garment['background'])
    if bounds is None:
        return None  # nobody in frame

    # Measured on the whole silhouette, not on the covered region: shorts are wider than tall
    # however squarely you stand, so the region's own shape says nothing about the pose.
    left, top, right, bottom = bounds
    if right - left >= bottom - top:
        return None  # foreshortened by leaning or turning, which no affine warp can undo

    target = class_map == garment['target']
    if not target.any():
        return None

    result = overlay(frame.copy(), garment['image'], target, garment['width'], garment['length'])
    for index in garment['occluders']:
        in_front = class_map == index
        result[in_front] = frame[in_front]
    return result
