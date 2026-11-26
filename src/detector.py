"""Object detection and tracking with a pre-trained YOLO model (CPU)."""
import time
import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="models/yolo26n-seg.pt", imgsz=640):
        """
        model_path: where the YOLO weights are
        imgsz:      internal image size for YOLO (multiple of 32: 320, 416, 640)
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz

        # The first call is always very slow (warm-up), so we do it now.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=self.imgsz, device="cpu", verbose=False)

    def detect(self, frame, conf=0.25, classes=None):
        """Detects objects in ONE frame (no memory between frames).
        Returns (detections, inference_ms)."""
        start = time.perf_counter()
        results = self.model.predict(
            frame, conf=conf, imgsz=self.imgsz,
            classes=classes, device="cpu", verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000
        return self._parse(results[0]), inference_ms

    def track(self, frame, conf=0.25, classes=None):
        """Detects AND tracks objects. Each detection gets a 'track_id'.
        persist=True tells YOLO to remember objects from the previous frame.
        Returns (detections, inference_ms)."""
        start = time.perf_counter()
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=conf, imgsz=self.imgsz, classes=classes,
            device="cpu", verbose=False,
        )
        inference_ms = (time.perf_counter() - start) * 1000
        return self._parse(results[0]), inference_ms

    def _parse(self, result):
        """Turns a YOLO result into a simple list of dictionaries."""
        # Mask outlines (one polygon per object); empty if no masks
        polygons = result.masks.xy if result.masks is not None else []

        # Track IDs exist only when tracking is used and objects are tracked
        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()

        detections = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])

            polygon = None
            if i < len(polygons) and len(polygons[i]) >= 3:
                polygon = np.array(polygons[i], dtype=np.int32)

            detections.append({
                "box": (x1, y1, x2, y2),
                "class_id": class_id,
                "name": result.names[class_id],
                "confidence": float(box.conf[0]),
                "polygon": polygon,
                "track_id": ids[i] if ids is not None else None,
            })
        return detections


    def reset_tracker(self):
        """Forget all tracked objects (call this before a new video)."""
        predictor = self.model.predictor
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()


def draw_detections(frame, detections):
    """Draws boxes, labels (with ID if tracked) on a COPY of the frame."""
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["name"]} {det["confidence"]:.2f}'
        if det.get("track_id") is not None:
            label = f'ID {det["track_id"]} {label}'

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - text_h - 6), (x1 + text_w, y1),
                      (0, 255, 0), -1)
        cv2.putText(output, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return output