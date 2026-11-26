# Real-Time Object Detection, Segmentation & Tracking

A lightweight computer vision pipeline that detects, outlines and tracks objects
in a webcam stream or an uploaded video, with a simple Streamlit interface.
Built to run on a **CPU-only laptop** (no GPU needed).

## Features
- Input: uploaded video file or webcam
- Gaussian blur and adaptive thresholding (OpenCV), each with an on/off toggle
- Object detection and instance segmentation with **YOLO26n-seg** (pre-trained, Ultralytics)
- Contour extraction with OpenCV (from the adaptive threshold, inside each detected box)
- Multi-object tracking with **ByteTrack** and unique IDs, plus movement trails
- Live per-class object counts and unique-ID count
- Performance monitoring: FPS, inference time, processing time, detected/tracked objects
  (saved to CSV, with a graph)

## How it works
```
Video -> resize -> YOLO26n-seg + ByteTrack (original frame) -> boxes, masks, IDs
                \-> Gaussian blur -> adaptive threshold -> contours
                         -> draw overlays + counts -> Streamlit UI + performance log
```
YOLO always receives the original frame. The blurred and thresholded images are
separate copies used only for contour extraction.

## Project structure
```
app.py                 Streamlit interface
benchmark.py           Fair speed tests on the same frames
src/
  video_input.py       Webcam / video file reading, resizing, frame skipping
  preprocessing.py     Gaussian blur and adaptive threshold
  detector.py          YOLO detection and tracking
  segmentation.py      Segmentation masks and OpenCV contours
  tracker.py           Movement trails and object counting
  performance.py       Measurements, CSV and graphs
  video_processor.py   Connects all steps for one frame
check_setup.py, test_*.py   Small test scripts used while building each phase
docs/                  Whitepaper, screenshots, measured results
```

## Installation (Windows, PowerShell)
Tested with Python 3.13.
```powershell
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
.\venv\Scripts\Activate.ps1

# CPU-only PyTorch (smaller download, no GPU needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Model weights download automatically on first run. Put `yolo26n-seg.pt` in the `models/` folder
(or run `python check_setup.py`, then move the downloaded file into `models/`).

## Run
```powershell
streamlit run app.py
```
Choose *Upload Video* or *Webcam* in the sidebar, adjust the settings, and press **Start**.

Note: changing a setting while a video is playing restarts it (normal Streamlit behavior).
Sample videos are not included; use your own `.mp4`.

## Performance
Measured on: Intel Core i7-3632QM, 8 GB RAM, Intel HD 4000 (CPU inference), Windows 10.

TODO: paste the table from `outputs/benchmark_summary.csv` here.

The pipeline is slow on this hardware (about 1.6 to 2 processed FPS at image size 640).
YOLO inference takes most of the time per frame.

## Limitations
- Low FPS on an old CPU; frame skipping is used to cover the video.
- Small, distant objects can be missed by the nano model (seen in an overhead crowd video).
- Adaptive-threshold contours also pick up background texture; YOLO masks are cleaner.
- Tracking IDs can switch when objects overlap or the frame skip is large.
- No ground-truth labels were used, so no precision/recall is reported.

## Future improvements
TODO: your own ideas (e.g. ONNX/OpenVINO export for faster CPU inference, labeled evaluation set).