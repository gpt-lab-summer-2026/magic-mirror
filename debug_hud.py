"""
Everything the `d` key draws, and the stopwatch behind it.

None of this changes the picture the demo shows. It lives here so the render
loop in main.py reads as the pipeline it is, instead of a pipeline with a
stopwatch wedged between every step.
"""
import time

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

timings = {}   # stage -> last frame's milliseconds
ages = {}      # what we composited with -> how old it was, in ms

HUD_COLOR = (0, 255, 255)
SKELETON_COLOR = (0, 255, 0)


def mark(stage, since):
    """Record ms since `since` and return a fresh mark for the next stage."""
    now = time.perf_counter()
    timings[stage] = (now - since) * 1000
    return now


def draw_skeleton(frame, all_pose_landmarks):
    """Pose landmarks and connections for every tracked body, drawn in place."""
    for pose_landmarks in all_pose_landmarks:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=SKELETON_COLOR, thickness=2))


def draw_hud(shown, frame_ms):
    """Stage costs, the frame total, and how stale what we drew with was."""
    # `stages` vs `total` is the gap: pose inference and everything else the
    # loop waits on but does not time. `parser` is outside the sum - it runs on
    # its own thread and costs this frame nothing.
    lines = [f"{stage} {ms:5.1f} ms" for stage, ms in timings.items()]
    lines.append(f"stages {sum(ms for s, ms in timings.items() if s != 'parser'):5.1f} ms")
    lines.append(f"total  {frame_ms:5.1f} ms   {1000 / frame_ms:4.1f} fps")
    # Staleness, which the stage costs cannot show: a starved pose thread still
    # submits in 2 ms, it just answers late.
    lines += [f"{what} age {ms:5.0f} ms" for what, ms in ages.items()]

    for i, line in enumerate(lines):
        cv2.putText(shown, line, (20, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2)


def dump_frame(clean, class_map, shown):
    """Freeze one frame to ignore/ for reading the artifact off disk afterwards.

    The class map is written raw - 0..17, near black to look at - so it can be
    inspected per class rather than guessed at from a colour ramp.
    """
    cv2.imwrite("ignore/debug_clean.png", clean)
    cv2.imwrite("ignore/debug_shown.png", shown)
    if class_map is not None:
        cv2.imwrite("ignore/debug_classes.png", class_map)
    print("wrote ignore/debug_clean.png, debug_shown.png, debug_classes.png", flush=True)
