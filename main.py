import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import camera
import composite
import config
import debug_hud
import display
import garment_overlay
import garment_overlay_bottom
import garment_rig
import garment_library
import parser_thread

# waitKeyEx codes for the arrows, Linux/Qt then Windows. They have no 8-bit
# form, which is why the loop reads waitKeyEx and never masks with 0xFF -
# 65361 & 0xFF is ord('Q').
KEY_LEFT = (65361, 2424832)
KEY_RIGHT = (65363, 2555904)

NAME_FLASH_FRAMES = 45   # ~1.5 s at 30 fps

latest_result = (None, 0)       # (pose result, timestamp of the frame it describes)
latest_result_lock = threading.Lock()


def store_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with latest_result_lock:
        latest_result = (result, timestamp_ms)


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=config.POSE_MODEL),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=store_result)

garment_library.load()
print(f"Loaded {len(garment_library.GARMENTS)} garments: "
      f"{', '.join(g.name for g in garment_library.GARMENTS)}", flush=True)

use_parser = parser_thread.available()
if use_parser:
    labels = parser_thread.start()
    for g in garment_library.GARMENTS:
        g.occluder_lut = composite.resolve_occluders(g, labels)
else:
    print("No GPU: parser off, so garments draw over hands and bare arms.", flush=True)

smoother = garment_rig.LandmarkSmoother(alpha=0.4)
frame_ms = 1000 / config.CAPTURE_FPS   # smoothed; the raw per-frame number is unreadable jitter
show_debug = False
fullscreen = config.ON_RIG
relight = True
name_frames = 0
frames = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = camera.open_camera()
    display.create_window(fullscreen)
    start_time = time.time()

    while cap.isOpened():
        loop_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        t = debug_hud.mark("decode", loop_start)
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
        debug_hud.ages["pose"] = now_ms - result_ms
        has_pose = result is not None and bool(result.pose_landmarks)

        if has_pose and show_debug:
            debug_hud.draw_skeleton(frame, result.pose_landmarks)
        # Submit cost, not inference: detect_async queues and returns, and the
        # model runs on MediaPipe's own thread. On 4 cores that thread competes
        # with this loop, so its real cost lands in `total`, not here.
        t = debug_hud.mark("pose", t)

        garment = garment_library.current()
        if has_pose and garment is not None:
            h, w = frame.shape[:2]
            # GarmentBottom needs hip/knee/ankle tracking and its own leg segments 
            overlay = (garment_overlay_bottom if isinstance(garment, garment_overlay_bottom.GarmentBottom)
                       else garment_overlay)
            body_points = overlay.get_body_points(result.pose_landmarks[0], w, h)
            if body_points is not None:
                frame = overlay.warp_and_blend(
                    frame, garment, smoother.update(body_points),
                    light_from=clean if relight else None)
                t = debug_hud.mark("warp", t)

                if use_parser:
                    if frames % config.PARSER_EVERY_N == 0:
                        # `clean` is a fresh copy every frame and nothing writes into
                        # it - warp_and_blend returns a new frame rather than mutating
                        # this one - so the worker can read it while the loop draws on.
                        parser_thread.submit(clean, now_ms)
                    class_map, class_map_ms = parser_thread.latest()
                    debug_hud.ages["map"] = now_ms - class_map_ms
                    # A map sized to a different frame would index out of bounds.
                    # Skipping the composite for one frame beats taking the kiosk down.
                    if class_map is not None and class_map.shape == frame.shape[:2]:
                        frame = composite.apply_occluders(frame, clean, class_map, garment.occluder_lut)
                    t = debug_hud.mark("occlude", t)

        shown = display.crop_to_display(frame)
        if name_frames > 0 and garment is not None:
            name_frames -= 1
            cv2.putText(shown, garment.name, (20, shown.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if show_debug:
            debug_hud.draw_hud(shown, frame_ms)
        debug_hud.draw_hint(shown)
        display.show(shown)
        debug_hud.mark("display", t)

        key = cv2.waitKeyEx(1)
        frame_ms = 0.6 * frame_ms + 0.4 * (time.perf_counter() - loop_start) * 1000
        if key == ord('q') or key == 27:
            break
        elif key in KEY_RIGHT:
            garment_library.next()
            name_frames = NAME_FLASH_FRAMES
        elif key in KEY_LEFT:
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
            debug_hud.dump_frame(clean, parser_thread.latest()[0], shown)
        elif key == ord('f'):
            fullscreen = not fullscreen
            display.apply_window_size(fullscreen)

    cap.release()
    cv2.destroyAllWindows()
