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
    return cv2.equalizeHist(gray)


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces and return bounding boxes as (x, y, w, h)."""
    normalized = normalize_frame_size(img_bgr)
    gray = _prepare_gray(normalized)
    faces = _CASCADE.detectMultiScale(gray, **_DETECT_PARAMS)
    if len(faces) == 0:
        return []

    scale = normalized.shape[1] / img_bgr.shape[1]
    if scale != 1.0:
        inv = 1.0 / scale
        faces = [
            (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            for (x, y, w, h) in faces
        ]

    return [tuple(map(int, face)) for face in faces]


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
    """Normalize, detect, and return the largest face crop from a frame."""
    normalized = normalize_frame_size(img_bgr)
    faces = detect_faces(img_bgr)
    if not faces:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Crop from normalized frame so pixel content matches live attendance frames.
    norm_scale = normalized.shape[1] / img_bgr.shape[1]
    if norm_scale != 1.0:
        x = int(x * norm_scale)
        y = int(y * norm_scale)
        w = int(w * norm_scale)
        h = int(h * norm_scale)
        return crop_face(normalized, x, y, w, h)

    return crop_face(img_bgr, x, y, w, h)
