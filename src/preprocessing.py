"""Image preprocessing: Gaussian Blur and Adaptive Thresholding."""
import cv2


def _make_odd(number, minimum=3):
    """Kernel and block sizes must be odd numbers."""
    number = max(minimum, int(number))
    return number if number % 2 == 1 else number + 1


def apply_gaussian_blur(frame, kernel_size=5):
    """Returns a blurred COPY of the frame. The original is not changed."""
    k = _make_odd(kernel_size)
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_adaptive_threshold(frame, block_size=11, c=2):
    """
    Returns a black-and-white image (objects = white).
    block_size: size of the neighborhood used for each pixel (odd number)
    c:          a value subtracted from the local average; higher = less noise
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    block = _make_odd(block_size)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, c,
    )


def preprocess(frame, use_blur=True, use_threshold=True,
               blur_kernel=5, block_size=11, c=2):
    """
    Runs the enabled steps on a copy of the frame.
    Returns (blurred_frame, threshold_image).
    threshold_image is None if thresholding is turned off.
    The original `frame` is never modified, so YOLO can still use it.
    """
    blurred = apply_gaussian_blur(frame, blur_kernel) if use_blur else frame
    threshold = None
    if use_threshold:
        threshold = apply_adaptive_threshold(blurred, block_size, c)
    return blurred, threshold