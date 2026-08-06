"""Save webcam frames as parser test input.

The parser has only ever seen 1080x1920 studio photos on white. This grabs the real
thing: whatever resolution the webcam gives, in whatever room you are standing in.

Shots are on a timer, because a full-body framing puts you too far away to reach a key.
"""

import time
from pathlib import Path

import cv2

OUTPUT_DIR = Path('webcam-captures')
INTERVAL = 8  # seconds: enough to walk back, and to change pose between shots


def next_path(directory):
    """Each capture is a different framing to compare, so none of them may overwrite another."""
    taken = len(list(directory.glob('capture_*.png')))
    return directory / f'capture_{taken + 1:02d}.png'


def draw_countdown(frame, seconds_left):
    """Sized to be readable from across the room, which is the whole reason the timer exists."""
    text = str(seconds_left)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
    origin = ((frame.shape[1] - text_width) // 2, (frame.shape[0] + text_height) // 2)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit('could not open the webcam')

OUTPUT_DIR.mkdir(exist_ok=True)
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'{int(width)}x{int(height)}   space = start/stop timer, q = quit')

next_shot = None

while True:
    success, frame = cap.read()
    if not success:
        raise SystemExit('dropped frame')

    preview = frame.copy()  # the countdown is drawn here so it stays out of the saved image
    if next_shot is not None:
        remaining = next_shot - time.time()
        if remaining <= 0:
            path = next_path(OUTPUT_DIR)
            cv2.imwrite(str(path), frame)
            print(f'saved {path}')
            next_shot = time.time() + INTERVAL
        else:
            draw_countdown(preview, int(remaining) + 1)

    cv2.imshow('capture', preview)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        next_shot = None if next_shot is not None else time.time() + INTERVAL

cap.release()
cv2.destroyAllWindows()
