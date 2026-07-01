import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_DETECT_PARAMS = {
    "scaleFactor": 1.08,
    "minNeighbors": 4,
    "minSize": (40, 40),
}

TARGET_FRAME_WIDTH = 640


def normalize_frame_size(img_bgr: np.ndarray, target_width: int = TARGET_FRAME_WIDTH) -> tuple[np.ndarray, float]:
    """Resize frame to consistent width for detection. Returns (normalized_frame, scale_factor)."""
    h, w = img_bgr.shape[:2]
    if w == target_width:
        return img_bgr, 1.0
    scale = target_width / w
    resized = cv2.resize(img_bgr, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply CLAHE contrast enhancement."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_faces_on_normalized(normalized: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces on normalized frame."""
    gray = _prepare_gray(normalized)
    faces = _CASCADE.detectMultiScale(gray, **_DETECT_PARAMS)
    return [tuple(map(int, face)) for face in faces]


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces and return bounding boxes as (x, y, w, h) in original image coordinates."""
    normalized, scale = normalize_frame_size(img_bgr)
    faces = _detect_faces_on_normalized(normalized)
    
    if not faces or scale == 1.0:
        return faces

    # Scale coordinates back to original image space
    inv_scale = 1.0 / scale
    faces = [(int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale)) for (x, y, w, h) in faces]

    return faces


def crop_face(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, padding: float = 0.15) -> np.ndarray:
    """Crop face region with padding."""
    img_h, img_w = img_bgr.shape[:2]
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img_bgr[y1:y2, x1:x2]


def extract_largest_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """Extract largest detected face from frame."""
    normalized, _ = normalize_frame_size(img_bgr)
    faces = _detect_faces_on_normalized(normalized)
    
    if not faces:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return crop_face(normalized, x, y, w, h)
