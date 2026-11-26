"""Connects all pipeline steps for ONE frame."""
import time
from src.preprocessing import preprocess
from src.detector import draw_detections
from src.segmentation import draw_segmentation
from src.tracker import TrackHistory, ObjectCounter, draw_counts
from src.performance import PerformanceMonitor


class VideoProcessor:
    def __init__(self, detector, conf=0.25, classes=None,
                 use_blur=True, use_threshold=True,
                 show_masks=True, show_contours=True):
        self.detector = detector
        self.conf = conf
        self.classes = classes
        self.use_blur = use_blur
        self.use_threshold = use_threshold
        self.show_masks = show_masks
        self.show_contours = show_contours

        # Fresh memory for every new video
        self.history = TrackHistory()
        self.counter = ObjectCounter()
        self.monitor = PerformanceMonitor()
        self.detector.reset_tracker()

    def process(self, frame):
        """Runs the full pipeline. Returns the drawn frame."""
        t_start = time.perf_counter()

        # YOLO gets the ORIGINAL frame; blur/threshold use a separate copy
        detections, inference_ms = self.detector.track(
            frame, conf=self.conf, classes=self.classes)
        _, threshold = preprocess(
            frame, use_blur=self.use_blur, use_threshold=self.use_threshold)

        self.history.update(detections)
        self.counter.update(detections)

        output = draw_segmentation(frame, detections, threshold,
                                   self.show_masks, self.show_contours)
        output = self.history.draw_trails(output)
        output = draw_detections(output, detections)
        output = draw_counts(output, self.counter)

        frame_ms = (time.perf_counter() - t_start) * 1000
        self.monitor.record(inference_ms, frame_ms,
                            detected=len(detections),
                            tracked=len(self.history.active_ids))
        return output