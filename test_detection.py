"""Phase 4 test: YOLO detection on video. Press q to quit."""
import cv2
from src.video_input import VideoInput
from src.detector import Detector, draw_detections

SOURCE = "sample_videos/test.mp4"   # or 0 for the webcam
IMGSZ = 640          # try 320, 416, 640
CONF = 0.25          # try 0.5
CLASSES = None       # None = all classes, [0] = people only
SKIP_FRAMES = 3      # process every 3rd frame
MAX_FRAMES = 40      # every test processes the same 40 frames

video = VideoInput(source=SOURCE, width=640, skip_frames=SKIP_FRAMES)
detector = Detector(imgsz=IMGSZ)

times = []
counts = []

while True:
    frame = video.read()
    if frame is None:
        break

    detections, inference_ms = detector.detect(frame, conf=CONF, classes=CLASSES)
    times.append(inference_ms)
    counts.append(len(detections))

    output = draw_detections(frame, detections)
    cv2.putText(output,
                f"Objects: {len(detections)}  Inference: {inference_ms:.0f} ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Phase 4 - Detection (press q to quit)", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

if times:
    print(f"imgsz={IMGSZ}, conf={CONF}, skip={SKIP_FRAMES}")
    print(f"Frames processed:        {len(times)}")
    print(f"Average inference time:  {sum(times) / len(times):.0f} ms")
    print(f"Average objects/frame:   {sum(counts) / len(counts):.1f}")