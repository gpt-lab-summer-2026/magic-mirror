"""
Machine constants: which camera, what it captures, which model file.

Geometry constants stay in garment_overlay.py next to the code that reads
them - a copy here would be a second source of truth, not a tidier one.
"""
import os

# /dev/videoN is handed out in plug order, so a reboot or a second camera can
# move the index with no error. This link always points at the demo camera.
CAMERA_BY_ID = "/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO._LTD_UHD_4K_Camera_N20200814003-video-index0"

# Used wherever that link does not resolve - any other machine, and Windows.
CAMERA_INDEX = 0

# The link resolves on the demo machine and nowhere else, so it answers both
# questions at once: which camera to open, and whether the screen is the
# rotated portrait one. They are the same machine.
RIG_DEVICE = os.path.realpath(CAMERA_BY_ID)
ON_RIG = RIG_DEVICE.startswith("/dev/video")

# 4:3, not 16:9: on a portrait screen the 9:16 crop keeps 540 of these 1280
# columns where 1280x720 would keep 405. MJPG only - YUYV tops out at 10fps
# at this size on a USB 2.0 bus.
CAPTURE_SIZE = (1280, 960)
CAPTURE_FPS = 30

POSE_MODEL = "pose_landmarker_full.task"

# Hand the parser thread every Nth frame. Raising this does not buy frame rate -
# the parse is off the render loop, so it only lowers the parser's duty cycle -
# and it costs staleness twice over: N frames of waiting on top of the parse
# itself. A stale map is worse than a lagging one, because the pixels copied are
# current while the mask is not: a moved arm punches its old shape through the
# garment. At 3 that measured ~120 ms and crossed arms tore visibly.
PARSER_EVERY_N = 1
GARMENT_DIR = "garments"   # scanned by garment_library.py in C2; nothing reads it yet
