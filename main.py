import os
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

import config
import garment_overlay
import garment_library

WINDOW = "Magic Mirror"

# The demo screen is a rotated monitor: a 16:9 frame letterboxed into it is
# mostly black bars, so the frame gets centre-cropped to this aspect instead.
DISPLAY_ASPECT = 9 / 16

# waitKeyEx codes for the arrows under Linux/Qt. They have no 8-bit form,
# which is why the loop reads waitKeyEx and never masks with 0xFF -
# 65361 & 0xFF is ord('Q').
KEY_LEFT = 65361
KEY_RIGHT = 65363

NAME_FLASH_FRAMES = 45   # ~1.5 s at 30 fps

latest_result = None
latest_result_lock = threading.Lock()


def store_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with latest_result_lock:
        latest_result = result


def open_camera():
    """Resolve the stable by-id link to whichever /dev/videoN it points at today."""
    device = os.path.realpath(config.CAMERA_BY_ID)
    if not device.startswith("/dev/video"):
        sys.exit(f"No camera at {config.CAMERA_BY_ID} - is it plugged in?")

    cap = cv2.VideoCapture(int(device.removeprefix("/dev/video")))
    if not cap.isOpened():
        sys.exit(f"Could not open {device}. The demo runs from the machine's own "
                 "desktop session - over SSH the camera is not reachable.")

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
    print(f"Camera {device}: {codec} {width}x{height} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps", flush=True)
    return cap


def crop_to_display(frame):
    """Centre-crop to the screen's aspect. Pose runs on the full frame, so an
    arm outside the crop keeps tracking - it just isn't shown."""
    h, w = frame.shape[:2]
    crop_w = int(h * DISPLAY_ASPECT)
    if crop_w >= w:
        return frame
    x0 = (w - crop_w) // 2
    return frame[:, x0:x0 + crop_w]


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=store_result)

garment_library.load()
print(f"Loaded {len(garment_library.GARMENTS)} garments: "
      f"{', '.join(g.name for g in garment_library.GARMENTS)}", flush=True)

smoother = garment_overlay.LandmarkSmoother(alpha=0.4)
show_debug = False
fullscreen = True
name_frames = 0

print("keys: <- -> garment   1-9 pick   d debug   f fullscreen   q / Esc quit", flush=True)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Mirror once, at the source: flip at display time instead and
        # MediaPipe's left_*/right_* describe the unflipped image while the
        # garment's anchors describe the flipped one.
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, int((time.time() - start_time) * 1000))

        with latest_result_lock:
            result = latest_result
        has_pose = result is not None and bool(result.pose_landmarks)

        if has_pose and show_debug:
            for pose_landmarks in result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))

        garment = garment_library.current()
        if has_pose and garment is not None:
            h, w = frame.shape[:2]
            body_points = garment_overlay.get_body_points(result.pose_landmarks[0], w, h)
            if body_points is not None:
                frame = garment_overlay.warp_and_blend(frame, garment, smoother.update(body_points))

        shown = crop_to_display(frame)
        if name_frames > 0 and garment is not None:
            name_frames -= 1
            cv2.putText(shown, garment.name, (20, shown.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(WINDOW, shown)

        key = cv2.waitKeyEx(1)
        if key == ord('q') or key == 27:
            break
        elif key == KEY_RIGHT:
            garment_library.next()
            name_frames = NAME_FLASH_FRAMES
        elif key == KEY_LEFT:
            garment_library.previous()
            name_frames = NAME_FLASH_FRAMES
        elif ord('1') <= key <= ord('9'):
            garment_library.select(key - ord('1'))
            name_frames = NAME_FLASH_FRAMES
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
