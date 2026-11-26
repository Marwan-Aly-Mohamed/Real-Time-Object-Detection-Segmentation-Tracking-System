"""Phase 6 test: detection + tracking IDs + movement trails."""
import cv2
from src.video_input import VideoInput
from src.detector import Detector, draw_detections
from src.tracker import TrackHistory

SOURCE = "sample_videos/test.mp4"
IMGSZ = 640
CONF = 0.25
SKIP_FRAMES = 2       # small skip = steadier tracking
MAX_FRAMES = 60       # stops by itself (keys are hard to catch when it's slow)

video = VideoInput(source=SOURCE, width=640, skip_frames=SKIP_FRAMES)
detector = Detector(imgsz=IMGSZ)
history = TrackHistory()
all_ids = set()
processed = 0

while processed < MAX_FRAMES:
    frame = video.read()
    if frame is None:
        break

    # NOTE: track() instead of detect()
    detections, inference_ms = detector.track(frame, conf=CONF)
    history.update(detections)
    all_ids.update(history.active_ids)
    processed += 1

    output = history.draw_trails(frame)
    output = draw_detections(output, detections)
    cv2.putText(output,
                f"Tracked now: {len(history.active_ids)}  "
                f"Unique IDs so far: {len(all_ids)}  {inference_ms:.0f} ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Phase 6 - Tracking (q to quit)", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print(f"Frames processed: {processed}")
print(f"Unique IDs created: {len(all_ids)}")