"""
A top's complete .anchors.json, read off its silhouette.

skeletonize() thins the cutout to a one-pixel tree: a sleeve becomes a branch,
its loose end the cuff and its middle the elbow, while the distance transform
under the spine gives the half-width the shoulders and hips come from. The
whole sidecar is returned because only the silhouette knows whether a garment
has sleeves, and sleeves - not its name - decide its segments and occluders.
"""
from collections import deque
from itertools import takewhile

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

ALPHA_CUTOFF = 127
WORK_HEIGHT = 512        # skeletonize here: at 1600px every contour bump grows its own twig
MIN_BLOB_SIZE = 0.05     # of image height - a smaller cutout is a speck, not a garment
MIN_ARM_LENGTH = 0.15    # of garment height - shorter branches are hem corners, not sleeves
MIN_ARM_REACH = 0.6      # of shoulder width - a cuff is out to the side, a hem tie hangs below
MIN_ARM_RADIUS = 0.02    # of garment height - thinner branches are ties and hanger wire
MAX_ARM_RADIUS = 0.5     # of the torso's half-width - a sleeve is always thinner than the body
MIN_TORSO_RADIUS = 0.15  # of the torso's half-width - thinner rows are ties and hanger wire
NECK_SHARE = 0.25        # of each half of the top edge - the inner end is collar, never shoulder
TORSO_LENGTH = 1.15      # shoulder line to hips, in shoulder widths
HANGING_ARM = (0.15, 1.5)  # no sleeve found: wrist offset from its shoulder, in shoulder widths
LONG_SLEEVE = 0.8   # of shoulder width - a cuff reaching this far from the shoulder is at the wrist
ARM_ELBOW = 0.55    # shoulder to elbow, in shoulder widths. Clicked: 0.53 0.53 0.56 0.61
ARM_WRIST = 1.15    # shoulder to wrist, in shoulder widths. Clicked: 1.03 1.06 1.20 1.35

# Proven: the mirror loads all three calibrated tops with these names on the rig.
BASE_OCCLUDERS = ["hands", "face", "hair"]


def silhouette(rgba):
    """The garment alone at working size, or (None, 0) if there is nothing to measure."""
    mask = (rgba[:, :, 3] > ALPHA_CUTOFF).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None, 0

    blob = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    smallest_side = min(stats[blob, cv2.CC_STAT_WIDTH], stats[blob, cv2.CC_STAT_HEIGHT])
    if smallest_side < MIN_BLOB_SIZE * labels.shape[0]:
        return None, 0

    scale = WORK_HEIGHT / labels.shape[0]
    mask = cv2.resize((labels == blob).astype(np.uint8), None,
                      fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # Lace holes would skeletonize into loops; the pad keeps an edge-touching mask off the array edge.
    return np.pad(binary_fill_holes(mask), 1), scale


def skeleton(mask):
    """The thinned tree, and how many skeleton neighbours each of its pixels has."""
    skel = skeletonize(mask).astype(np.uint8)
    neighbours = cv2.filter2D(skel, -1, np.ones((3, 3), np.uint8)) - skel
    return skel.astype(bool), neighbours * skel


def paths_to_ends(skel, neighbours, seed):
    """The route from every loose end of the skeleton back to the torso."""
    # Breadth-first from the torso, not walk-until-fork: a hem frill forks one sleeve a dozen times.
    previous = {seed: None}
    queue = deque([seed])
    while queue:
        y, x = queue.popleft()
        for step in ((y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1),
                     (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)):
            if skel[step] and step not in previous:
                previous[step] = (y, x)
                queue.append(step)

    paths = []
    for end in zip(*np.nonzero(neighbours == 1)):
        path, point = [], end
        while point is not None:
            path.append(point)
            point = previous.get(point)
        paths.append(path)      # loose end first, torso last
    return paths


def torso_profile(skel, mask):
    """Per row: which column the garment is thickest in, and how thick it is there."""
    # Wisp rows dropped: a dress's waist ties hang below its hem at a radius of one pixel.
    radius = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) * skel
    thickest = radius.max(axis=1)
    rows = np.flatnonzero(thickest >= MIN_TORSO_RADIUS * thickest.max())
    centers = radius[rows].argmax(axis=1)
    return rows, centers, radius[rows, centers], radius


def garment_run(mask, row, column):
    """How far the garment reaches either side of one point, along one row."""
    left = right = column
    while mask[row, left - 1]:
        left -= 1
    while mask[row, right + 1]:
        right += 1
    return left, right


def find_arms(paths, radius, height, seed, shoulder_width):
    """The (left, right) sleeves as thin runs walked in from each cuff, None per side without one."""
    # Where the cuff sits is what really tells a sleeve from a hem tie, and it is not close:
    # cuffs reach 0.74-0.84 shoulder widths out from the middle and everything else 0.47 or less.
    # Length carried that decision alone once, and picked a sleeve by one pixel of branch.
    sleeves = []
    for path in paths:
        thin = list(takewhile(lambda p: radius[p] < MAX_ARM_RADIUS * radius.max(), path))
        if len(thin) < MIN_ARM_LENGTH * height:
            continue
        if abs(thin[0][1] - seed[1]) < MIN_ARM_REACH * shoulder_width:
            continue
        outer = np.mean([radius[p] for p in thin[:len(thin) // 2]])
        if outer >= MIN_ARM_RADIUS * height:
            sleeves.append(thin)

    left = [s for s in sleeves if s[0][1] < seed[1]]
    right = [s for s in sleeves if s[0][1] > seed[1]]
    return (min(left, key=lambda s: s[0][1], default=None),
            max(right, key=lambda s: s[0][1], default=None))


def top_edge(mask):
    """The first opaque row in each column - the garment's top outline, as a curve."""
    cols = np.flatnonzero(mask.any(0))
    return cols, np.argmax(mask[:, cols], axis=0)


def shoulder_points(mask):
    """Per side, the corner of the top edge: where its climb from the cuff flattens off."""
    # A corner, not an extreme, so a shoulder under a collar and a 1px strap both survive.
    cols, top = top_edge(mask)
    middle = len(cols) // 2
    # left_* is the image's left, not the wearer's: photographed front-up, a garment has its
    # right sleeve there. Screen-side is the convention throughout - calibrate.py, the clicked
    # sidecars, and the mirrored frame main.py feeds the pose model - and swapping it here
    # alone would not mirror the garment but shrink it, a similarity fit having no reflection.
    halves = [(cols[:middle + 1], top[:middle + 1]),        # left half, cuff first
              (cols[middle:][::-1], top[middle:][::-1])]    # right half, cuff first
    corners = []
    for half_cols, half_top in halves:
        x, y = half_cols.astype(float), half_top.astype(float)
        line = y[0] + (x - x[0]) * (y[-1] - y[0]) / (x[-1] - x[0])
        # Cuff side only: a camp collar rises as far above the chord as the shoulder
        # does, and when the two tie it is noise that picks, leaving the rig tilted.
        corner = int(np.argmax((line - y)[:int((1 - NECK_SHARE) * len(x))]))
        corners.append((x[corner], y[corner]))
    return corners


def spine(rows, centers, radius, seed_row):
    """The torso's own stretch of the profile, up and down from its thickest row."""
    # It ends where the thickest point jumps sideways - the profile landing on a sleeve.
    top = bottom = int(np.searchsorted(rows, seed_row))
    while top > 0 and abs(centers[top - 1] - centers[top]) <= radius[top]:
        top -= 1
    while bottom + 1 < len(rows) and abs(centers[bottom + 1] - centers[bottom]) <= radius[bottom]:
        bottom += 1
    return top, bottom


def torso_top(rows, radius):
    """Where the garment's body begins: its first row at half full width."""
    # Not the shoulder: a strap carries that 240px above the body on black_dress.
    return int(rows[np.argmax(radius >= 0.5 * radius.max())])


def hip_row(torso_y, shoulder_width, hem_y):
    """One shoulder width below the top of the body, or the hem where the garment ends first."""
    return int(min(torso_y + TORSO_LENGTH * shoulder_width, hem_y))


def along(start, through, distance):
    """The point `distance` away from start, on the ray through `through`."""
    step = (through[0] - start[0], through[1] - start[1])
    reach = np.hypot(*step)
    return (start[0] + step[0] / reach * distance, start[1] + step[1] / reach * distance)


def hanging_wrist(shoulder, width, side):
    """No sleeve to follow, so the arm hangs: straight down, angled slightly out."""
    out, down = HANGING_ARM
    sign = -1 if side == "left" else 1
    return (shoulder[0] + sign * out * width, shoulder[1] + down * width)


def top_anchors(rgba):
    """The complete sidecar in source pixels, or None when the silhouette cannot be measured."""
    mask, scale = silhouette(rgba)
    if mask is None:
        return None

    skel, neighbours = skeleton(mask)
    rows, centers, radius, radius_map = torso_profile(skel, mask)
    # Off the mask, not off the profile: the profile drops the collar and the hem as too thin,
    # which on one garment left height at 65% of the real thing and every ratio below that tight.
    mask_rows = np.flatnonzero(mask.any(1))
    height = mask_rows[-1] - mask_rows[0]

    # The thickest skeleton pixel is the middle of the torso: every path walks out from there.
    seed = np.unravel_index(radius_map.argmax(), radius_map.shape)

    (left_x, left_y), (right_x, right_y) = shoulder_points(mask)
    arms = find_arms(paths_to_ends(skel, neighbours, seed), radius_map, height, seed,
                     right_x - left_x)

    def at(row):
        i = int(np.searchsorted(rows, row).clip(0, len(rows) - 1))
        return centers[i], radius[i]

    def torso_run(row):
        # Widths off the mask: where a sleeve runs alongside the body the thickest pixel is its own.
        column = seed[1] if mask[row, seed[1]] else at(row)[0]
        return garment_run(mask, row, column)

    _, hem = spine(rows, centers, radius, seed[0])
    hip_y = hip_row(torso_top(rows, radius), right_x - left_x, rows[hem])
    hip_left, hip_right = torso_run(hip_y)
    points = {
        "left_shoulder": (left_x, left_y),
        "right_shoulder": (right_x, right_y),
        "hip_center": ((hip_left + hip_right) / 2, hip_y),
    }
    width = right_x - left_x
    to_the_wrist = []
    for side, arm in zip(("left", "right"), arms):
        shoulder = points[f"{side}_shoulder"]
        if arm is None:
            wrist = hanging_wrist(shoulder, width, side)
            elbow = ((shoulder[0] + wrist[0]) / 2, (shoulder[1] + wrist[1]) / 2)
        else:
            cuff = (arm[0][1], arm[0][0])
            if np.hypot(cuff[0] - shoulder[0], cuff[1] - shoulder[1]) >= LONG_SLEEVE * width:
                to_the_wrist.append(side)
                wrist = cuff
                elbow = (arm[len(arm) // 2][1], arm[len(arm) // 2][0])
            else:
                # A short sleeve stops on the upper arm, so the joints are further out
                # along the arm it lies on. Reading them off the fabric instead puts a
                # whole elbow and wrist inside one short sleeve, and the rig then
                # stretches that sleeve down the wearer's arm to reach them.
                elbow = along(shoulder, cuff, ARM_ELBOW * width)
                wrist = along(shoulder, cuff, ARM_WRIST * width)
        points[f"{side}_wrist"] = wrist
        points[f"{side}_elbow"] = elbow

    measured = [side for side, arm in zip(("left", "right"), arms) if arm is not None]
    segments = ["torso"]
    for side in measured:
        segments.append(f"{side}_upper_arm")
        # No forearm segment where there is no forearm fabric: it would carry sleeve
        # down the bare arm. The upper arm segment keeps the whole sleeve either way.
        if side in to_the_wrist:
            segments.append(f"{side}_forearm")

    sidecar = {name: [int((x - 1) / scale), int((y - 1) / scale)]
               for name, (x, y) in points.items()}
    sidecar["segments"] = segments
    # One sleeve still counts as sleeved: repainting both arms would erase the sleeve there is.
    sidecar["occluders"] = BASE_OCCLUDERS if measured else BASE_OCCLUDERS + ["arms"]
    return sidecar
