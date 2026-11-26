"""Real-Time Object Detection & Tracking - Streamlit app."""
import os
import tempfile
import streamlit as st

from src.video_input import VideoInput
from src.detector import Detector
from src.video_processor import VideoProcessor
from src.tracker import display_name

st.set_page_config(page_title="Real-Time Object Detection & Tracking",
                   layout="wide")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model (the first time takes a while)...")
def load_detector(imgsz):
    """Loads the model once and reuses it (cached by image size)."""
    return Detector(model_path="models/yolo26n-seg.pt", imgsz=imgsz)


def save_upload(uploaded_file):
    """OpenCV needs a real file path, so we save the upload to a temp file."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
    return tmp.name


# The "notebook" that survives Streamlit reruns
defaults = {"running": False, "processor": None, "last_frames": None,
            "upload_key": None, "upload_path": None}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.header("Settings")

source_choice = st.sidebar.radio("Input source", ["Upload Video", "Webcam"])
uploaded = None
if source_choice == "Upload Video":
    uploaded = st.sidebar.file_uploader(
        "Video file", type=["mp4", "avi", "mov", "mkv"])

conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)

use_blur = st.sidebar.toggle("Gaussian Blur", value=True)
use_threshold = st.sidebar.toggle("Adaptive Thresholding", value=True)
show_masks = st.sidebar.toggle("Show segmentation masks", value=True)
show_contours = st.sidebar.toggle(
    "Show OpenCV contours (needs thresholding)", value=True)

st.sidebar.subheader("Speed settings")
# 640 finds more small objects; 416 is faster. Default depends on the source.
imgsz = st.sidebar.selectbox(
    "YOLO image size", [320, 416, 640],
    index=2 if source_choice == "Upload Video" else 1)
skip_frames = st.sidebar.slider("Process every Nth frame", 1, 10, 2)
width = st.sidebar.selectbox("Frame width (pixels)", [320, 480, 640], index=2)

detector = load_detector(imgsz)

# Class selection (COCO class names come from the model)
all_names = list(detector.model.names.values())
default_classes = [n for n in ["person", "car", "bicycle", "dog", "cat"]
                   if n in all_names]
selected_names = st.sidebar.multiselect(
    "Object classes (empty = all)", all_names, default=default_classes)
name_to_id = {name: i for i, name in detector.model.names.items()}
class_ids = [name_to_id[n] for n in selected_names] or None

start_clicked = st.sidebar.button("Start", type="primary")
stop_clicked = st.sidebar.button("Stop")
if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False

# ---------------------------------------------------------------
# Main page layout
# ---------------------------------------------------------------
st.title("Real-Time Object Detection & Tracking")
st.write(
    "This app detects objects with a lightweight YOLO model, outlines them "
    "with segmentation masks and OpenCV contours, tracks them with unique IDs, "
    "and shows live counts and performance numbers. "
    "Choose a source in the sidebar and press **Start**."
)

col_original, col_processed = st.columns(2)
col_original.subheader("Original")
col_processed.subheader("Processed")
original_ph = col_original.empty()
processed_ph = col_processed.empty()

st.subheader("Performance")
m1, m2, m3, m4 = st.columns(4)
fps_ph, inf_ph, det_ph, trk_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()

st.subheader("Live object counts")
counts_ph = st.empty()


def show_stats(processor):
    """Fills the metric cards and the counts list from the latest values."""
    rows = processor.monitor.rows
    if not rows:
        return
    last = rows[-1]
    fps_ph.metric("Current FPS", f"{processor.monitor.current_fps():.1f}")
    inf_ph.metric("Inference Time", f"{last['inference_ms']:.0f} ms")
    det_ph.metric("Detected Objects", last["detected"])
    trk_ph.metric("Active Tracked Objects", last["tracked"])

    counter = processor.counter
    ordered = sorted(counter.current.items(), key=lambda item: -item[1])
    lines = [f"- **{display_name(name)}:** {count}" for name, count in ordered]
    lines.append(f"- **Total Objects:** {counter.total_current()}")
    lines.append(f"- **Unique IDs seen:** {counter.total_unique()}")
    counts_ph.markdown("\n".join(lines))


# ---------------------------------------------------------------
# Decide what to process
# ---------------------------------------------------------------
source = None
if st.session_state.running:
    if source_choice == "Webcam":
        source = 0
    elif uploaded is None:
        st.warning("Please upload a video file first.")
        st.session_state.running = False
    else:
        # Save the upload only once (not on every rerun)
        key = (uploaded.name, uploaded.size)
        if st.session_state.upload_key != key:
            st.session_state.upload_key = key
            st.session_state.upload_path = save_upload(uploaded)
        source = st.session_state.upload_path

# ---------------------------------------------------------------
# The live processing loop
# ---------------------------------------------------------------
if st.session_state.running and source is not None:
    try:
        video = VideoInput(source=source, width=width, skip_frames=skip_frames)
    except ValueError as error:
        st.error(str(error))
        st.session_state.running = False
        video = None

    if video is not None:
        processor = VideoProcessor(
            detector, conf=conf, classes=class_ids,
            use_blur=use_blur, use_threshold=use_threshold,
            show_masks=show_masks, show_contours=show_contours)
        st.session_state.processor = processor

        try:
            while True:
                frame = video.read()
                if frame is None:          # video finished
                    break
                output = processor.process(frame)

                original_ph.image(frame, channels="BGR")
                processed_ph.image(output, channels="BGR")
                show_stats(processor)
                st.session_state.last_frames = (frame, output)
        finally:
            video.release()                # always free the camera/file

        # We only get here if the video ended by itself.
        # (Stop / setting changes interrupt the script before this line.)
        st.session_state.running = False
        st.success("Video finished.")

# ---------------------------------------------------------------
# After stopping: show the last frame and the results
# ---------------------------------------------------------------
processor = st.session_state.processor
if processor is not None and not st.session_state.running:
    if st.session_state.last_frames is not None:
        last_original, last_output = st.session_state.last_frames
        original_ph.image(last_original, channels="BGR")
        processed_ph.image(last_output, channels="BGR")
    show_stats(processor)

    if processor.monitor.rows:
        st.subheader("Performance results")
        st.write(processor.monitor.summary())
        st.pyplot(processor.monitor.make_plot())
        st.download_button(
            "Download measurements (CSV)",
            data=processor.monitor.to_dataframe().to_csv(index=False),
            file_name="performance.csv", mime="text/csv")
elif processor is None and not st.session_state.running:
    st.info("Waiting for input. Choose a source and press Start.")