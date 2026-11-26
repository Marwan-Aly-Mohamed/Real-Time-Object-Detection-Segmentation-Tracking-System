"""Phase 5 test: detection + YOLO masks + OpenCV contours (step mode).
Click the window first, then use:
  SPACE = next frame   m = masks on/off   c = contours on/off   q = quit
"""
import cv2
from src.video_input import VideoInput
from src.preprocessing import preprocess
from src.detector import Detector, draw_detections
from src.segmentation import draw_segmentation

SOURCE = "sample_videos/test.mp4"
IMGSZ = 640
CONF = 0.25
SKIP_FRAMES = 10     # bigger jump between frames, so each SPACE shows something new

video = VideoInput(source=SOURCE, width=640, skip_frames=SKIP_FRAMES)
detector = Detector(imgsz=IMGSZ)
show_masks = True
show_contours = True
need_new_frame = True

while True:
    # YOLO runs ONLY when we ask for a new frame
    if need_new_frame:
        frame = video.read()
        if frame is None:
            break
        detections, inference_ms = detector.detect(frame, conf=CONF)
        _, threshold = preprocess(frame, use_blur=True, use_threshold=True)
        need_new_frame = False

    # Drawing is fast, so toggles update instantly
    output = draw_segmentation(frame, detections, threshold,
                               show_masks, show_contours)
    output = draw_detections(output, detections)
    cv2.putText(output,
                f"Objects: {len(detections)}  Inference: {inference_ms:.0f} ms  "
                f"Masks: {show_masks}  Contours: {show_contours}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Phase 5 - Segmentation (SPACE / m / c / q)", output)

    # waitKey(0) = wait here until a key is pressed (no more missed keys)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        show_masks = not show_masks
    elif key == ord("c"):
        show_contours = not show_contours
    elif key == ord(" "):
        need_new_frame = True

video.release()
cv2.destroyAllWindows()