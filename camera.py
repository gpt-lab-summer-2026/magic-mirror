"""
Opening the camera: the rig's if this is the rig, otherwise whatever the
machine has. The capture mode is only forced on the rig - those settings are
facts about that one USB camera, not about cameras in general.
"""
import sys

import cv2

import config


def open_camera():
    """The rig's camera when its by-id link resolves, else config.CAMERA_INDEX."""
    index = (int(config.RIG_DEVICE.removeprefix("/dev/video"))
             if config.ON_RIG else config.CAMERA_INDEX)

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. The demo runs from the machine's own "
                 "desktop session - over SSH the camera is not reachable.")

    if config.ON_RIG:
        # FOURCC first: it sets the bandwidth budget, and only MJPG fits a full
        # frame on this USB 2.0 bus. Set it after the size and the driver has
        # already picked a size for the format it was previously in.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
    # 2, not 1: with a single buffer the driver has nowhere to put frame N+1
    # while we hold frame N, and the capture rate halves - measured 15 vs 30.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # A driver silently substitutes a mode it does support, so print what we
    # actually got rather than what we asked for.
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {index}: {codec} {width}x{height} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps", flush=True)
    return cap
