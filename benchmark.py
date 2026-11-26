"""Benchmark: runs the pipeline on the SAME frames with different settings
and saves real measurements to outputs/."""
import os
import pandas as pd
from src.video_input import VideoInput
from src.detector import Detector
from src.video_processor import VideoProcessor

VIDEO = "sample_videos/test.mp4"
FRAMES = 40      # every run processes exactly these many frames
SKIP = 2
REPEATS = 2      # each setting is run twice, then averaged
CONF = 0.25

CONFIGS = [
    {"name": "seg-320",            "model": "models/yolo26n-seg.pt", "imgsz": 320, "pre": True},
    {"name": "seg-416",            "model": "models/yolo26n-seg.pt", "imgsz": 416, "pre": True},
    {"name": "seg-640",            "model": "models/yolo26n-seg.pt", "imgsz": 640, "pre": True},
    {"name": "seg-640-no-preproc", "model": "models/yolo26n-seg.pt", "imgsz": 640, "pre": False},
    {"name": "detect-only-640",    "model": "models/yolo26n.pt",     "imgsz": 640, "pre": True},
]


def run_once(detector, pre):
    video = VideoInput(VIDEO, width=640, skip_frames=SKIP)
    processor = VideoProcessor(detector, conf=CONF,
                               use_blur=pre, use_threshold=pre)
    for _ in range(FRAMES):
        frame = video.read()
        if frame is None:
            break
        processor.process(frame)
    video.release()
    return processor.monitor.summary()


rows = []
for cfg in CONFIGS:
    if not os.path.exists(cfg["model"]):
        print(f"Skipping {cfg['name']}: {cfg['model']} not found")
        continue
    print(f"Running {cfg['name']} ...")
    detector = Detector(model_path=cfg["model"], imgsz=cfg["imgsz"])
    for run in range(REPEATS):
        summary = run_once(detector, cfg["pre"])
        rows.append({"config": cfg["name"], "run": run + 1, **summary})
        print(f"  run {run + 1}: {summary['avg_fps']} FPS, "
              f"{summary['avg_inference_ms']} ms inference")

df = pd.DataFrame(rows)
df.to_csv("outputs/benchmark_runs.csv", index=False)

average = (df.groupby("config", sort=False)
             [["avg_fps", "avg_inference_ms", "avg_frame_ms", "avg_detected"]]
             .mean().round(2))
average.to_csv("outputs/benchmark_summary.csv")

print("\n--- Benchmark summary (average of runs) ---")
print(average)
print("\nSaved: outputs/benchmark_runs.csv and outputs/benchmark_summary.csv")