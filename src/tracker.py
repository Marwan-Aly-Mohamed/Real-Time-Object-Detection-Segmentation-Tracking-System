"""Keeps the movement history (trails) of tracked objects."""
from collections import deque
import cv2
import numpy as np


def color_for_id(track_id):
    """Gives each ID its own (always visible) color."""
    return (50 + (track_id * 67) % 200,
            50 + (track_id * 131) % 200,
            50 + (track_id * 29) % 200)


class TrackHistory:
    def __init__(self, max_points=30, max_missing=15):
        """
        max_points:  how many past positions to keep per object (trail length)
        max_missing: forget an object after it was not seen for this many
                     processed frames (it left the frame or was lost)
        """
        self.max_points = max_points
        self.max_missing = max_missing
        self.trails = {}        # id -> deque of (x, y) center points
        self.last_seen = {}     # id -> frame number when last seen
        self.frame_number = 0
        self.active_ids = set() # IDs visible in the current frame

    def update(self, detections):
        """Call once per processed frame with the tracked detections."""
        self.frame_number += 1
        self.active_ids = set()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = det["box"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id not in self.trails:
                self.trails[track_id] = deque(maxlen=self.max_points)
            self.trails[track_id].append(center)
            self.last_seen[track_id] = self.frame_number
            self.active_ids.add(track_id)

        # Forget objects that disappeared a while ago
        for track_id in list(self.last_seen):
            if self.frame_number - self.last_seen[track_id] > self.max_missing:
                del self.last_seen[track_id]
                del self.trails[track_id]

    def draw_trails(self, frame):
        """Draws the path of each visible object on a COPY of the frame."""
        output = frame.copy()
        for track_id in self.active_ids:
            points = np.array(self.trails[track_id], dtype=np.int32)
            if len(points) >= 2:
                cv2.polylines(output, [points], False,
                              color_for_id(track_id), 2)
        return output

# ---------------------------------------------------------------
# Counting
# ---------------------------------------------------------------
def display_name(name):
    """Nicer names for the on-screen text."""
    return {"person": "People"}.get(name, name.capitalize())


class ObjectCounter:
    def __init__(self):
        self.current = {}        # class name -> how many are visible right now
        self.id_to_class = {}    # track id -> class name (every ID ever seen)

    def update(self, detections):
        """Call once per processed frame."""
        self.current = {}
        for det in detections:
            name = det["name"]
            self.current[name] = self.current.get(name, 0) + 1
            track_id = det.get("track_id")
            if track_id is not None:
                self.id_to_class[track_id] = name

    def total_current(self):
        return sum(self.current.values())

    def total_unique(self):
        return len(self.id_to_class)

    def unique_counts(self):
        counts = {}
        for name in self.id_to_class.values():
            counts[name] = counts.get(name, 0) + 1
        return counts


def draw_counts(frame, counter):
    """Draws the live counts in the top-left corner of a COPY of the frame."""
    output = frame.copy()

    # Most common classes first; keep at most 5 so the box stays small
    ordered = sorted(counter.current.items(), key=lambda item: -item[1])[:5]
    lines = [f"{display_name(name)}: {count}" for name, count in ordered]
    lines.append(f"Total Objects: {counter.total_current()}")
    lines.append(f"Unique IDs seen: {counter.total_unique()}")

    line_height = 22
    box_height = line_height * len(lines) + 10

    # Dark see-through background so the text is readable
    overlay = output.copy()
    cv2.rectangle(overlay, (5, 5), (235, 5 + box_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

    for i, text in enumerate(lines):
        cv2.putText(output, text, (12, 28 + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return output