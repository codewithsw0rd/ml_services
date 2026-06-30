"""
Shared face detection helpers used by registration and attendance endpoints.
"""
import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Live webcam frames are noisier than uploaded registration photos.
_LIVE_DETECT_PARAMS = {
    "scaleFactor": 1.08,
    "minNeighbors": 3,
    "minSize": (36, 36),
}

_STATIC_DETECT_PARAMS = {
    "scaleFactor": 1.1,
    "minNeighbors": 5,
    "minSize": (60, 60),
}


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + histogram equalization improves Haar detection in poor lighting."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def _maybe_upscale(img_bgr: np.ndarray, min_width: int = 640) -> np.ndarray:
    """Upscale small webcam frames so Haar can find faces more reliably."""
    h, w = img_bgr.shape[:2]
    if w >= min_width:
        return img_bgr
    scale = min_width / w
    return cv2.resize(
        img_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def detect_faces(img_bgr: np.ndarray, live: bool = False) -> list[tuple[int, int, int, int]]:
    """
    Detect frontal faces and return bounding boxes as (x, y, w, h) in image coords.
    """
    prepared = _maybe_upscale(img_bgr) if live else img_bgr
    scale = prepared.shape[1] / img_bgr.shape[1]
    gray = _prepare_gray(prepared)
    params = _LIVE_DETECT_PARAMS if live else _STATIC_DETECT_PARAMS

    faces = _CASCADE.detectMultiScale(gray, **params)
    if len(faces) == 0:
        return []

    if scale != 1.0:
        inv = 1.0 / scale
        faces = [
            (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            for (x, y, w, h) in faces
        ]

    return [tuple(map(int, face)) for face in faces]


def crop_face(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, padding: float = 0.15) -> np.ndarray:
    """
    Crop a face region with proportional padding.
    Padding helps HOG capture hairline/chin context that Haar boxes often clip.
    """
    img_h, img_w = img_bgr.shape[:2]
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img_bgr[y1:y2, x1:x2]
