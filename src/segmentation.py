"""Segmentation masks (from YOLO) and contour extraction (from OpenCV)."""
import cv2
import numpy as np


def contours_in_box(threshold, box, min_area=30):
    """
    Finds the biggest white shape inside a detection box
    of the adaptive-threshold image.
    Returns a contour in full-frame coordinates, or None.
    """
    x1, y1, x2, y2 = box
    h, w = threshold.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = threshold[y1:y2, x1:x2]          # crop the box area
    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    # Move the contour from crop coordinates back to full-frame coordinates
    return largest + np.array([x1, y1], dtype=np.int32)


def draw_segmentation(frame, detections, threshold=None,
                      show_masks=True, show_contours=True):
    """
    Draws YOLO masks (blue) and OpenCV contours (yellow) on a COPY of the frame.
    """
    output = frame.copy()

    if show_masks:
        overlay = frame.copy()
        for det in detections:
            poly = det.get("polygon")
            if poly is not None:
                cv2.fillPoly(overlay, [poly], (255, 0, 0))
        # Mix 35% colored overlay with 65% original = see-through fill
        output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)
        for det in detections:
            poly = det.get("polygon")
            if poly is not None:
                cv2.polylines(output, [poly], True, (255, 0, 0), 2)

    if show_contours and threshold is not None:
        for det in detections:
            contour = contours_in_box(threshold, det["box"])
            if contour is not None:
                cv2.drawContours(output, [contour], -1, (0, 255, 255), 1)

    return output