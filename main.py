import threading
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import cv2
import numpy as np
import bg_remove

model_path = 'pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
latest_result_lock = threading.Lock()

test_image = 'test-images/bg_white_top.png'

def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)
  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in detection_result.pose_landmarks:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


# Create a pose landmarker instance with the live stream mode:
def store_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with latest_result_lock:
        latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=store_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  # Use OpenCV's VideoCapture to start capturing from the webcam.
  cap = cv2.VideoCapture(0)
  start_time = time.time()

  # Create a loop to read the latest frame from the camera using VideoCapture#read()
  while cap.isOpened():
    success, numpy_frame_from_opencv = cap.read()
    if not success:
      print('Ignoring empty camera frame.')
      break

    numpy_frame_from_opencv = cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_BGR2RGB)

    # Convert the frame received from OpenCV to a MediaPipe's Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

    frame_timestamp_ms = int((time.time() - start_time) * 1000)

    # Send live image data to perform pose landmarking.
    # The results are accessible via the `result_callback` provided in
    # the `PoseLandmarkerOptions` object.
    # The pose landmarker must be created with the live stream mode.
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    with latest_result_lock:
      result_to_draw = latest_result

    if result_to_draw is not None and result_to_draw.pose_landmarks:
      annotated_frame = draw_landmarks_on_image(numpy_frame_from_opencv, result_to_draw)
    else:
      annotated_frame = numpy_frame_from_opencv

    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.putText(display_frame, "press i to process image and q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam', display_frame)

    # Check for user input for the test image or quit the application
    if cv2.waitKey(1) & 0xFF == ord('i'):
      print("Removing background from test image...")
      no_bg_image_path = bg_remove.remove_background_from_image(test_image, 'test-images-output')
      print(f"Background removed image saved at: {no_bg_image_path}")
          
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
