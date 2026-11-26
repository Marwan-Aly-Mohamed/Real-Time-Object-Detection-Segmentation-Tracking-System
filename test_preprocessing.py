"""Phase 3 test: see Original | Blurred | Adaptive Threshold side by side.
Keys:  b = blur on/off   t = threshold on/off   q = quit
"""
import cv2
import numpy as np
from src.video_input import VideoInput
from src.preprocessing import preprocess

SOURCE = "sample_videos/test.mp4"   # or 0 for the webcam

video = VideoInput(source=SOURCE, width=400)
use_blur = True
use_threshold = True

if video.is_webcam or video.source_fps <= 0:
    delay = 1
else:
    delay = int(1000 / video.source_fps)

while True:
    frame = video.read()
    if frame is None:
        break

    blurred, threshold = preprocess(frame, use_blur, use_threshold)

    # Threshold is 1-channel; convert to 3 channels so we can place it next to the others
    if threshold is None:
        threshold_view = np.zeros_like(frame)
    else:
        threshold_view = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([frame, blurred, threshold_view])
    cv2.putText(combined, f"Blur: {use_blur}  Threshold: {use_threshold}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Original | Blurred | Adaptive Threshold", combined)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        use_blur = not use_blur
    elif key == ord("t"):
        use_threshold = not use_threshold

video.release()
cv2.destroyAllWindows()