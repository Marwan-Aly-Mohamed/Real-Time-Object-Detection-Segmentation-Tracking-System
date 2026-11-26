"""Phase 7 test: full pipeline with counts and performance measurement."""
import time
import cv2
from src.video_input import VideoInput
from src.preprocessing import preprocess
from src.detector import Detector, draw_detections
from src.segmentation import draw_segmentation
from src.tracker import TrackHistory, ObjectCounter, draw_counts
from src.performance import PerformanceMonitor

SOURCE = "sample_videos/test.mp4"
IMGSZ = 640
CONF = 0.25
SKIP_FRAMES = 2
MAX_FRAMES = 60

video = VideoInput(source=SOURCE, width=640, skip_frames=SKIP_FRAMES)
detector = Detector(imgsz=IMGSZ)
history = TrackHistory()
counter = ObjectCounter()
monitor = PerformanceMonitor()
processed = 0

while processed < MAX_FRAMES:
    t_start = time.perf_counter()          # start timing the WHOLE frame

    frame = video.read()
    if frame is None:
        break

    # YOLO gets the original frame; blur/threshold use a separate copy
    detections, inference_ms = detector.track(frame, conf=CONF)
    _, threshold = preprocess(frame, use_blur=True, use_threshold=True)

    history.update(detections)
    counter.update(detections)

    output = draw_segmentation(frame, detections, threshold)
    output = history.draw_trails(output)
    output = draw_detections(output, detections)
    output = draw_counts(output, counter)

    frame_ms = (time.perf_counter() - t_start) * 1000   # stop timing
    monitor.record(inference_ms, frame_ms,
                   detected=len(detections),
                   tracked=len(history.active_ids))
    processed += 1

    cv2.putText(output,
                f"FPS: {monitor.current_fps():.1f}  Inference: {inference_ms:.0f} ms",
                (10, output.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Phase 7 - Full pipeline (q to quit)", output)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# Save the real measurements
monitor.save_csv("outputs/performance.csv")
monitor.save_plot("outputs/performance.png")

print("\n--- Performance summary ---")
for key, value in monitor.summary().items():
    print(f"{key}: {value}")
print("Unique counts by class:", counter.unique_counts())
print("Saved: outputs/performance.csv and outputs/performance.png")