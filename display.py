"""
The window and the shape of what goes in it.

The frame is cropped to DISPLAY_ASPECT once, then letterboxed into whatever
size the window currently is - which starts at DISPLAY_ASPECT but the user
can drag to anything. WINDOW_KEEPRATIO is 0 in this build and the Win32
backend has no letterbox of its own, so show() does the letterboxing by hand
every frame rather than relying on the window shape matching the image.
"""
import cv2
import numpy as np

WINDOW = "Magic Mirror"

# The demo screen is a rotated monitor: a 16:9 frame letterboxed into it is
# mostly black bars, so the frame gets centre-cropped to this aspect instead.
DISPLAY_ASPECT = 9 / 16

# Height of the desktop window at startup and after leaving fullscreen.
WINDOW_HEIGHT = 900


def create_window(fullscreen):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    apply_window_size(fullscreen)


def apply_window_size(fullscreen):
    """Fullscreen, or a portrait window the same shape as the cropped frame.
    The user can still drag either one to another shape afterwards - show()
    letterboxes to whatever the window ends up being."""
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


def _letterbox(frame, target_w, target_h):
    """Scale frame to fit inside target_w x target_h without distorting it,
    padding the leftover area with black bars."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def show(frame):
    # (x, y, w, h) of the window's client area - w/h is 0 before the window is
    # first mapped, and Windows sometimes reports it as such transiently.
    _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW)
    if win_w <= 0 or win_h <= 0:
        cv2.imshow(WINDOW, frame)
        return
    cv2.imshow(WINDOW, _letterbox(frame, win_w, win_h))
