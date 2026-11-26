"""Phase 2 test: show webcam or video frames in a window."""
import time
import cv2
from src.video_input import VideoInput

# CHANGE THIS: 0 = webcam, or a path like "sample_videos/test.mp4"
SOURCE = "D:\\Projects\\Real-Time Object Detection, Segmentation & Tracking System\\sample_videos\\test.mp4"

video = VideoInput(source=SOURCE, width=640, skip_frames=1)
print(f"Video FPS: {video.source_fps}, total frames: {video.total_frames}")

last_time = time.time()

while True:
    frame = video.read()
    if frame is None:
        print("Video ended.")
        break

    # Measure how many frames per second we are showing
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Phase 2 - Video Input (press q to quit)", frame)

    # Files: wait one frame's duration (25 FPS -> 40 ms). Webcam: just 1 ms.
    if video.is_webcam or video.source_fps <= 0:
        delay = 1
    else:
        delay = int(1000 / video.source_fps)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()