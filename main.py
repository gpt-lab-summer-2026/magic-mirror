import os
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

import composite
import config
import garment_overlay
import garment_library
import human_parser

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

latest_result = (None, 0)       # (pose result, timestamp of the frame it describes)
latest_result_lock = threading.Lock()

latest_class_map = (None, 0)    # (class map, timestamp of the frame it describes)
latest_class_map_lock = threading.Lock()
parser_frame = None        # (frame, timestamp) waiting for the worker
parser_frame_lock = threading.Lock()

timings = {}   # stage -> last frame's milliseconds, drawn by `d`
ages = {}      # what we composited with -> how old it was, in ms


def store_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with latest_result_lock:
        latest_result = (result, timestamp_ms)


def parse_worker():
    """Latest frame wins: a frame handed over while this is busy replaces the
    pending one rather than joining a queue, so the map is always the freshest
    the parser could have finished - never a backlog of stale ones.

    The only thread that touches the torch model.
    """
    global latest_class_map, parser_frame
    while True:
        with parser_frame_lock:
            pending, parser_frame = parser_frame, None
        if pending is None:
            time.sleep(0.005)   # nothing pending; spinning here would cost a core
            continue
        frame, stamp = pending
        start = time.perf_counter()
        class_map = human_parser.parse(parser_model, parser_processor, frame)
        mark("parser", start)
        with latest_class_map_lock:
            latest_class_map = (class_map, stamp)


def mark(stage, since):
    """Record ms since `since` and return a fresh mark for the next stage."""
    now = time.perf_counter()
    timings[stage] = (now - since) * 1000
    return now


def dump_debug(clean, class_map, shown):
    """Freeze one frame to ignore/ for reading the artifact off disk afterwards.

    The class map is written raw - 0..17, near black to look at - so it can be
    inspected per class rather than guessed at from a colour ramp.
    """
    cv2.imwrite("ignore/debug_clean.png", clean)
    cv2.imwrite("ignore/debug_shown.png", shown)
    if class_map is not None:
        cv2.imwrite("ignore/debug_classes.png", class_map)
    print("wrote ignore/debug_clean.png, debug_shown.png, debug_classes.png", flush=True)


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

parser_model, parser_processor = human_parser.load_parser()
labels = human_parser.class_indices(parser_model)
print(f"Parser on {parser_model.device}, classes: {', '.join(sorted(labels))}", flush=True)
for g in garment_library.GARMENTS:
    g.occluder_lut = composite.resolve_occluders(g, labels)

# daemon: `q` must end the process even if the parser is mid-inference. The
# worker holds no file or socket and its class map is throwaway, so there is
# nothing a clean shutdown would protect - and a join() on a wedged torch call
# is exactly the kiosk that needs Ctrl-C.
threading.Thread(target=parse_worker, daemon=True).start()

smoother = garment_overlay.LandmarkSmoother(alpha=0.4)
frame_ms = 1000 / config.CAPTURE_FPS   # smoothed; the raw per-frame number is unreadable jitter
show_debug = False
fullscreen = True
relight = True
name_frames = 0
frames = 0

print("keys: <- -> garment   1-9 pick   d debug   l relight   s dump frame   "
      "f fullscreen   q / Esc quit", flush=True)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start_time = time.time()

    while cap.isOpened():
        loop_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        t = mark("decode", loop_start)
        frames += 1

        # Mirror once, at the source: flip at display time instead and
        # MediaPipe's left_*/right_* describe the unflipped image while the
        # garment's anchors describe the flipped one.
        frame = cv2.flip(frame, 1)
        # The parser and the occluder repaint must both see the camera and not
        # the debug skeleton, or `d` bakes green lines into the arms.
        clean = frame.copy()

        now_ms = int((time.time() - start_time) * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, now_ms)

        with latest_result_lock:
            result, result_ms = latest_result
        ages["pose"] = now_ms - result_ms
        has_pose = result is not None and bool(result.pose_landmarks)

        if has_pose and show_debug:
            for pose_landmarks in result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
        # Submit cost, not inference: detect_async queues and returns, and the
        # model runs on MediaPipe's own thread. On 4 cores that thread competes
        # with this loop, so its real cost lands in `total`, not here.
        t = mark("pose", t)

        garment = garment_library.current()
        if has_pose and garment is not None:
            h, w = frame.shape[:2]
            body_points = garment_overlay.get_body_points(result.pose_landmarks[0], w, h)
            if body_points is not None:
                frame = garment_overlay.warp_and_blend(
                    frame, garment, smoother.update(body_points),
                    light_from=clean if relight else None)
                t = mark("warp", t)

                if frames % config.PARSER_EVERY_N == 0:
                    # `clean` is a fresh copy every frame and nothing writes into
                    # it - warp_and_blend returns a new frame rather than mutating
                    # this one - so the worker can read it while the loop draws on.
                    with parser_frame_lock:
                        parser_frame = (clean, now_ms)
                with latest_class_map_lock:
                    class_map, class_map_ms = latest_class_map
                ages["map"] = now_ms - class_map_ms
                # A map sized to a different frame would index out of bounds.
                # Skipping the composite for one frame beats taking the kiosk down.
                if class_map is not None and class_map.shape == frame.shape[:2]:
                    frame = composite.apply_occluders(frame, clean, class_map, garment.occluder_lut)
                t = mark("occlude", t)

        shown = crop_to_display(frame)
        if name_frames > 0 and garment is not None:
            name_frames -= 1
            cv2.putText(shown, garment.name, (20, shown.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if show_debug:
            # `stages` vs `total` is the gap: pose inference and everything else
            # this loop waits on but does not time. `parser` is outside the sum -
            # it runs on its own thread and costs this frame nothing.
            hud = [f"{stage} {ms:5.1f} ms" for stage, ms in timings.items()]
            hud.append(f"stages {sum(ms for s, ms in timings.items() if s != 'parser'):5.1f} ms")
            hud.append(f"total  {frame_ms:5.1f} ms   {1000 / frame_ms:4.1f} fps")
            # How stale what we drew with was, which the stage costs cannot show:
            # a starved pose thread still submits in 2 ms, it just answers late.
            hud += [f"{what} age {ms:5.0f} ms" for what, ms in ages.items()]
            for i, line in enumerate(hud):
                cv2.putText(shown, line, (20, 30 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW, shown)
        mark("display", t)

        key = cv2.waitKeyEx(1)
        frame_ms = 0.6 * frame_ms + 0.4 * (time.perf_counter() - loop_start) * 1000
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
        elif key == ord('l'):
            relight = not relight
            print(f"relight {'on' if relight else 'off'}", flush=True)
        elif key == ord('s'):
            dump_debug(clean, latest_class_map[0], shown)
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
