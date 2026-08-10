"""
The window and the shape of what goes in it.

Two aspects have to agree here: the crop the frame is cut to, and the window
that crop is painted into. They are both derived from DISPLAY_ASPECT, because
nothing in OpenCV enforces the match - see apply_window_size.
"""
import cv2

WINDOW = "Magic Mirror"

# The demo screen is a rotated monitor: a 16:9 frame letterboxed into it is
# mostly black bars, so the frame gets centre-cropped to this aspect instead.
DISPLAY_ASPECT = 9 / 16

# Height of the desktop window. Fullscreen anywhere but the rig would stretch
# the 9:16 crop to the screen's aspect: WINDOW_KEEPRATIO is 0 in this build and
# the Win32 backend has no letterbox, so a WINDOW_NORMAL window always scales
# the image to fill it. Sizing the window to the image is what keeps it 9:16.
WINDOW_HEIGHT = 900


def create_window(fullscreen):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    apply_window_size(fullscreen)


def apply_window_size(fullscreen):
    """Fullscreen, or a portrait window the same shape as the cropped frame."""
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    if not fullscreen:
        cv2.resizeWindow(WINDOW, round(WINDOW_HEIGHT * DISPLAY_ASPECT), WINDOW_HEIGHT)


def crop_to_display(frame):
    """Centre-crop to the screen's aspect. Pose runs on the full frame, so an
    arm outside the crop keeps tracking - it just isn't shown."""
    h, w = frame.shape[:2]
    crop_w = int(h * DISPLAY_ASPECT)
    if crop_w >= w:
        return frame
    x0 = (w - crop_w) // 2
    return frame[:, x0:x0 + crop_w]


def show(frame):
    cv2.imshow(WINDOW, frame)
