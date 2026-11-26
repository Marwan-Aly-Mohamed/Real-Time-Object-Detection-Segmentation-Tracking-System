"""Collects real performance measurements (FPS, inference time, counts)."""
import os
import pandas as pd
from matplotlib.figure import Figure


class PerformanceMonitor:
    def __init__(self):
        self.rows = []

    def record(self, inference_ms, frame_ms, detected, tracked):
        """
        Call once per processed frame.
        inference_ms: time YOLO took
        frame_ms:     time the whole pipeline took for this frame
        detected:     number of detected objects
        tracked:      number of tracked objects (with an ID)
        """
        fps = 1000 / frame_ms if frame_ms > 0 else 0
        self.rows.append({
            "frame": len(self.rows) + 1,
            "inference_ms": round(inference_ms, 1),
            "frame_ms": round(frame_ms, 1),
            "fps": round(fps, 2),
            "detected": detected,
            "tracked": tracked,
        })

    def current_fps(self, window=5):
        """Average FPS of the last few frames (less jumpy than one frame)."""
        if not self.rows:
            return 0.0
        recent = self.rows[-window:]
        return sum(r["fps"] for r in recent) / len(recent)

    def to_dataframe(self):
        return pd.DataFrame(self.rows)

    def summary(self):
        """Average / min / max values, or {} if nothing was recorded."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        return {
            "frames": len(df),
            "avg_fps": round(df["fps"].mean(), 2),
            "avg_inference_ms": round(df["inference_ms"].mean(), 1),
            "min_inference_ms": round(df["inference_ms"].min(), 1),
            "max_inference_ms": round(df["inference_ms"].max(), 1),
            "avg_frame_ms": round(df["frame_ms"].mean(), 1),
            "avg_detected": round(df["detected"].mean(), 1),
            "avg_tracked": round(df["tracked"].mean(), 1),
        }

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def make_plot(self):
        """Returns a matplotlib figure: FPS and inference time per frame."""
        df = self.to_dataframe()
        fig = Figure(figsize=(8, 5))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        ax1.plot(df["frame"], df["fps"], color="tab:blue")
        ax1.set_ylabel("FPS")
        ax1.set_title("Processing FPS per frame")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["frame"], df["inference_ms"], color="tab:red")
        ax2.set_ylabel("Inference time (ms)")
        ax2.set_xlabel("Processed frame")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def save_plot(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.make_plot().savefig(path, dpi=120)