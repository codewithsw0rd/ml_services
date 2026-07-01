"""
Shared face detection helpers using Haar Cascade with edge detection + morphological preprocessing.

Algorithms from Computer Science Syllabus:
- Edge Detection: Canny edge detector
- Morphological Operations: Erosion, dilation for noise reduction
- Face Detection: Haar Cascade Classifier (Viola-Jones)
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

# ── Morphological kernel for edge cleanup ────────────────────────────────────
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


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


def _prepare_gray_with_edge_enhancement(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess grayscale image with edge detection + morphology for better face detection.
    
    Pipeline:
    1. Grayscale conversion
    2. CLAHE for contrast enhancement
    3. Canny edge detection
    4. Morphological operations (dilation → erosion) to clean edges
    5. Return edge-enhanced grayscale for Haar Cascade
    
    This helps Haar Cascade find faces in challenging conditions:
    - Poor lighting (CLAHE handles this)
    - Cluttered backgrounds (edges isolate face structure)
    - Noise (morphology cleans spurious edges)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # ── Contrast enhancement ─────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # ── Edge detection (Canny) ───────────────────────────────────────────
    # Compute image gradients to find face boundaries
    edges = cv2.Canny(clahe_img, threshold1=50, threshold2=150)
    
    # ── Morphological operations for edge cleanup ────────────────────────
    # Dilation: thickens edges to bridge small gaps
    dilated = cv2.dilate(edges, MORPH_KERNEL, iterations=2)
    
    # Erosion: removes small noise/isolated pixels
    cleaned_edges = cv2.erode(dilated, MORPH_KERNEL, iterations=1)
    
    # ── Combine original CLAHE with cleaned edge map ──────────────────────
    # Blend edge information back to enhance facial features
    # Edges get higher weight where they're strong, but don't override base image
    combined = cv2.addWeighted(clahe_img, 0.7, cleaned_edges, 0.3)
    
    return combined


def _prepare_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Original preprocessing (for backward compatibility if needed)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_faces_on_normalized(normalized: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces on an already-normalized frame using enhanced preprocessing."""
    # Use edge-enhanced preprocessing for better detection
    gray = _prepare_gray_with_edge_enhancement(normalized)
    faces = _CASCADE.detectMultiScale(gray, **_DETECT_PARAMS)
    if len(faces) == 0:
        return []
    return [tuple(map(int, face)) for face in faces]


def detect_faces(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces and return bounding boxes as (x, y, w, h).

    Uses Canny edge detection + morphological operations for robust preprocessing.
    
    Pipeline:
    1. Normalize frame to TARGET_FRAME_WIDTH
    2. Apply CLAHE contrast enhancement
    3. Compute Canny edges
    4. Apply morphological operations (dilate → erode)
    5. Blend edges with original for Haar Cascade
    6. Detect faces with Haar Cascade
    
    Returns coordinates in the *original* image's pixel space.
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

    Uses edge-enhanced detection for more reliable face finding.
    """
    normalized = normalize_frame_size(img_bgr)
    faces = _detect_faces_on_normalized(normalized)
    if not faces:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return crop_face(normalized, x, y, w, h)
