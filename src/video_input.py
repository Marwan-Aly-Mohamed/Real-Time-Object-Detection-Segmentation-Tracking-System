"""Reads frames from a webcam or a video file."""
import cv2


class VideoInput:
    def __init__(self, source=0, width=640, skip_frames=1):
        """
        source:      0 for the webcam, or a path to a video file (string)
        width:       every frame is resized to this width (height follows)
        skip_frames: 1 = use every frame, 2 = every 2nd frame, and so on
        """
        self.is_webcam = isinstance(source, int)

        if self.is_webcam:
            # CAP_DSHOW makes the webcam open faster on Windows
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # <-- new line
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self.width = width
        self.skip_frames = max(1, skip_frames)
        self.frame_count = 0
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        """Returns the next resized frame, or None when the video ends."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_count += 1

        # Skip frames: grab() moves forward without fully decoding the frame
        for _ in range(self.skip_frames - 1):
            if not self.cap.grab():
                break

        return self._resize(frame)

    def _resize(self, frame):
        """Shrinks the frame to self.width, keeping the aspect ratio."""
        height, width = frame.shape[:2]
        if width == self.width:
            return frame
        scale = self.width / width
        new_size = (self.width, int(height * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def release(self):
        """Closes the webcam or file."""
        self.cap.release()