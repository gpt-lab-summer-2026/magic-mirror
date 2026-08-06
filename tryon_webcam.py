"""Live try-on: parse once on a keypress, then composite every frame against that mask.

Parsing costs seconds on the CPU, so the mask cannot follow the body. The garment stays
where you were standing when you pressed space, and drifts as you move. Watching that
break is the point of this script.
"""

from pathlib import Path

import cv2

from composite import apply_garment, prepare_garment
from human_parser import load_parser, parse, class_indices

# Each garment: the cutout, the class it covers, how much wider and longer than that region it
# hangs, and the parts redrawn over it afterwards. A sleeved garment must not list 'arms', or
# its sleeves are painted over with bare arm the moment they are drawn.
GARMENTS = [
    {'path': 'test-images-output/w_black_dress.out.png', 'target': 'torso',
     'width': 1.3, 'length': 1.7, 'occluders': ['hands', 'arms', 'face', 'hair']},
    {'path': 'test-images-output/w_pink_skirt.out.png', 'target': 'pants',
     'width': 1.3, 'length': 1.3, 'occluders': ['hands', 'arms', 'face', 'hair']},
    {'path': 'test-images-output/w_white_top.out.png', 'target': 'torso',
     'width': 2.5, 'length': 1.1, 'occluders': ['hands', 'face', 'hair']},
]


def draw_status(frame, text):
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit('could not open the webcam')

print('loading parser...')
model, processor, device = load_parser()
index = class_indices(model)
garments = [prepare_garment(record, index) for record in GARMENTS]
names = [Path(record['path']).stem for record in GARMENTS]
print('space = fit, n = next garment, q = quit')

class_map = None
chosen = 0

while True:
    success, frame = cap.read()
    if not success:
        raise SystemExit('dropped frame')

    display, status = frame.copy(), f'{names[chosen]} - press space to fit'
    if class_map is not None:
        result = apply_garment(frame, class_map, garments[chosen])
        display, status = (frame.copy(), f'{names[chosen]} - stand facing the mirror') \
            if result is None else (result, names[chosen])

    draw_status(display, status)
    cv2.imshow('magic mirror', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('n'):
        chosen = (chosen + 1) % len(garments)
    if key == ord(' '):
        busy = frame.copy()
        draw_status(busy, 'parsing...')
        cv2.imshow('magic mirror', busy)
        cv2.waitKey(1)  # force a repaint, because the parse below freezes the window
        class_map = parse(model, processor, frame, device)

cap.release()
cv2.destroyAllWindows()
