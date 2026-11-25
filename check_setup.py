"""Phase 1 check: make sure OpenCV, PyTorch and YOLO all work."""
import time
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

print("OpenCV version:     ", cv2.__version__)
print("PyTorch version:    ", torch.__version__)
print("Ultralytics version:", ultralytics.__version__)
print("CUDA available:     ", torch.cuda.is_available(), "(False is expected on your laptop)")

# Load the nano segmentation model (downloads automatically the first time)
model = YOLO("yolo26n-seg.pt")

# Make a blank test image so we don't need any file
test_image = np.zeros((480, 640, 3), dtype=np.uint8)

start = time.time()
results = model(test_image, device="cpu", verbose=False)
elapsed_ms = (time.time() - start) * 1000

print(f"Model loaded and ran one inference in {elapsed_ms:.0f} ms")
print("Setup is OK!")