import cv2
import numpy as np
from skimage.feature import hog

TARGET_SIZE = (100, 100)

HOG_ORIENTATION = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR/RGB face crop into a normalised 1-D feature vector.

    Pipeline:
        1. Grayscale conversion     — removes colour channel noise
        2. Resize to TARGET_SIZE   — fixed input dimensionality
        3. CLAHE equalisation      — handles uneven lighting
        4. HOG feature extraction  — encodes local gradient structure
        5. L2 normalisation        — puts all vectors on the unit sphere
                                     so Euclidean ≈ cosine distance

    Args:
        img: H×W×C uint8 array (BGR from OpenCV, or RGB — both work
             because we convert to grayscale immediately).

    Returns:
        1-D float64 numpy array  (length depends on HOG params;
        with the defaults above on a 100×100 image: 1764 dims).
    """
    
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(resized)
    
    feature_vec = hog(
        equalised,
        orientation=HOG_ORIENTATION,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    
    norm = np.linalg.norm(feature_vec)
    if norm > 0:
        feature_vec = feature_vec / norm
    
    return feature_vec


def calculate_face_quality(face_crop: np.ndarray) -> float:
    """
    Calculate face image quality score (0-1).
    Evaluates sharpness, contrast, and brightness of the face region.
    
    Quality metrics:
    1. Sharpness (Laplacian variance) - detects blur
    2. Contrast (histogram spread) - detects under/over-exposure
    3. Brightness (mean intensity) - ensures not too dark/bright
    
    Args:
        face_crop: H×W×C uint8 array of face region from image
    
    Returns:
        quality_score: float 0-1, where 1 is perfect quality
    """
    
    # Convert to grayscale if needed
    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop.copy()
    
    h, w = gray.shape
    
    # ── 1. Sharpness Score (Laplacian variance) ───────────────────────
    # Measures edge definition; higher variance = sharper image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: typically ranges 0-5000, cap at 2500 for score
    sharpness_score = min(laplacian_var / 2500.0, 1.0)
    
    # ── 2. Contrast Score (histogram spread) ──────────────────────────
    # Measures intensity distribution; good balance = higher score
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    # Good contrast: std > 20, centered around 128
    contrast_score = (std_intensity / 128.0) * 0.5  # Max 0.5 from std
    brightness_score = 1.0 - abs(mean_intensity - 128.0) / 128.0  # Max 0.5 from centering
    brightness_score = max(brightness_score * 0.5, 0.0)
    
    # ── 3. Face Size Penalty ─────────────────────────────────────────
    # Very small faces are harder to recognize; penalize if too small
    face_area = h * w
    min_area = 60 * 60  # Minimum 60×60 pixels
    optimal_area = 200 * 200  # Optimal is larger
    
    if face_area < min_area:
        size_score = 0.3
    elif face_area < optimal_area:
        # Linear interpolation: small→0.3 to optimal→1.0
        size_score = 0.3 + (0.7 * (face_area - min_area) / (optimal_area - min_area))
    else:
        # Penalize if too large (fills entire frame)
        max_area = 400 * 400
        size_score = 1.0 - min(0.5 * (face_area - optimal_area) / (max_area - optimal_area), 0.5)
    
    # ── Final score: weighted combination ─────────────────────────────
    # Sharpness: 50%, Contrast+Brightness: 30%, Size: 20%
    quality_score = (
        sharpness_score * 0.50 +
        (contrast_score + brightness_score) * 0.30 +
        size_score * 0.20
    )
    
    # Clamp to 0-1 range and round
    quality_score = max(0.0, min(1.0, quality_score))
    
    return round(quality_score, 4)