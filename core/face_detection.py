"""
Shared face detection helpers used by registration and attendance endpoints.
"""
import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# One profile for registration + live attendance so embeddings stay comparable.
_DETECT_PARAMS = {
    "scaleFactor": 1.08,
    "minNeighbors": 4,
    "minSize": (40, 40),
}

TARGET_FRAME_WIDTH = 640


def normalize_frame_size(img_bgr: np.ndarray, target_width: int = TARGET_FRAME_WIDTH) -> np.ndarray:
    """Resize incoming frames to the same width used by the live attendance webcam."""
    h, w = img_bgr.shape[:2]
    if w == target_width:
        return img_bgr
    scale = target_width / w
    return cv2.resize(
        img_bgr,
        (target_width, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Use CLAHE to match the preprocessing pipeline used for HOG embedding extraction.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_faces_on_normalized(normalized: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces on an already-normalized frame (no resize done here)."""
    gray = _prepare_gray(normalized)
    faces = _CASCADE.detectMultiScale(gray, **_DETECT_PARAMS)
    if len(faces) == 0:
        return []
    return [tuple(map(int, face)) for face in faces]


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces and return bounding boxes as (x, y, w, h).

    Normalizes the frame to TARGET_FRAME_WIDTH before detection so that
    the cascade parameters are consistent regardless of input resolution.
    Returned coordinates are in the *original* image's pixel space.
    """
    normalized = normalize_frame_size(img_bgr)
    faces = _detect_faces_on_normalized(normalized)
    if not faces:
        return []

    # Scale bounding boxes back to the original image coordinate space.
    scale = normalized.shape[1] / img_bgr.shape[1]
    if scale != 1.0:
        inv = 1.0 / scale
        faces = [
            (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            for (x, y, w, h) in faces
        ]

    return faces


def crop_face(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, padding: float = 0.15) -> np.ndarray:
    """Crop a face region with proportional padding."""
    img_h, img_w = img_bgr.shape[:2]
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img_bgr[y1:y2, x1:x2]


def extract_largest_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """Normalize, detect, and return the largest face crop from a frame.

    Always crops from the *normalized* (640 px wide) frame so that the pixel
    content of the crop matches what the live attendance stream produces —
    keeping enrollment and live-detection embeddings comparable.
    """
    # Normalize once; all subsequent work is done on this frame.
    normalized = normalize_frame_size(img_bgr)
    faces = _detect_faces_on_normalized(normalized)
    if not faces:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return crop_face(normalized, x, y, w, h)
