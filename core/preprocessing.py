import cv2
import numpy as np
from skimage.feature import hog

TARGET_SIZE = (100, 100)

HOG_ORIENTATION = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ── Morphological kernel for edge enhancement ────────────────────────────────
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _extract_edge_features(img: np.ndarray) -> np.ndarray:
    """
    Extract edge-based features using Canny edge detection + Sobel gradients.
    
    Algorithms from Computer Science Syllabus:
    - Canny Edge Detection: finds face contours and structure
    - Sobel Operators: computes directional gradients
    - Morphological Operations: cleans edge map
    
    Returns 1-D feature vector from edge map that complements HOG.
    """
    # Canny edge detection to find face structure
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    
    # Morphological cleanup: close small gaps, remove noise
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)
    
    # Sobel gradients for directional information
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude of gradient
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Combine edges with gradient magnitude for richer features
    # Edges: structural info, Gradients: smooth transition info
    combined = (cleaned_edges / 255.0) * 0.5 + (gradient_mag / gradient_mag.max()) * 0.5
    
    # Convert to 1-D feature vector
    edge_features = combined.flatten()
    
    # Normalize to unit vector
    norm = np.linalg.norm(edge_features)
    if norm > 0:
        edge_features = edge_features / norm
    
    return edge_features


def prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR/RGB face crop into a normalised 1-D feature vector.

    Pipeline:
        1. Grayscale conversion     — removes colour channel noise
        2. Resize to TARGET_SIZE   — fixed input dimensionality
        3. CLAHE equalisation      — handles uneven lighting
        4. HOG feature extraction  — encodes local gradient structure
        5. Canny edge detection    — captures face structure/contours
        6. Morphology operations   — cleans edge artifacts
        7. Combine HOG + edge features — richer descriptor
        8. L2 normalisation        — puts all vectors on the unit sphere
                                     so Euclidean ≈ cosine distance

    Args:
        img: H×W×C uint8 array (BGR from OpenCV, or RGB — both work
             because we convert to grayscale immediately).

    Returns:
        1-D float64 numpy array with combined HOG + edge features.
        Shape: (2544,) — 1764 from HOG + 780 from edges
    """
    
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # ── Standard preprocessing ──────────────────────────────────────────
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(resized)
    
    # Mild denoise before feature extraction
    denoised = cv2.bilateralFilter(equalised, d=5, sigmaColor=50, sigmaSpace=50)
    
    # ── HOG features (original) ─────────────────────────────────────────
    hog_features = hog(
        denoised,
        orientations=HOG_ORIENTATION,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )
    
    # ── Edge-based features (new) ───────────────────────────────────────
    edge_features = _extract_edge_features(denoised)
    
    # ── Combine both feature types ──────────────────────────────────────
    # HOG captures local texture and gradients
    # Edges capture global face structure and contours
    # Weighted combination: HOG (70%) + Edges (30%) for robust matching
    combined_features = np.concatenate([
        hog_features * 0.7,
        edge_features * 0.3
    ])
    
    # ── Final L2 normalization ──────────────────────────────────────────
    norm = np.linalg.norm(combined_features)
    if norm > 0:
        combined_features = combined_features / norm
    
    return combined_features


def prepare_image_flip_pair(img: np.ndarray) -> list[np.ndarray]:
    """Original + horizontally flipped embeddings for webcam mirror tolerance.
    
    Uses improved feature extraction with edge detection.
    """
    return [prepare_image(img), prepare_image(cv2.flip(img, 1))]


def calculate_face_quality(face_crop: np.ndarray) -> float:
    """
    Calculate face image quality score (0-1).
    
    Evaluates:
    1. Sharpness (Laplacian variance) - detects blur
    2. Contrast (histogram spread) - detects under/over-exposure
    3. Brightness (mean intensity) - ensures not too dark/bright
    4. Edge density (new) - measures well-defined facial features
    
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
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 2500.0, 1.0)
    
    # ── 2. Contrast Score (histogram spread) ──────────────────────────
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    contrast_score = (std_intensity / 128.0) * 0.5
    brightness_score = 1.0 - abs(mean_intensity - 128.0) / 128.0
    brightness_score = max(brightness_score * 0.5, 0.0)
    
    # ── 3. Face Size Penalty ─────────────────────────────────────────
    face_area = h * w
    min_area = 60 * 60
    optimal_area = 200 * 200
    
    if face_area < min_area:
        size_score = 0.3
    elif face_area < optimal_area:
        size_score = 0.3 + (0.7 * (face_area - min_area) / (optimal_area - min_area))
    else:
        max_area = 400 * 400
        size_score = 1.0 - min(0.5 * (face_area - optimal_area) / (max_area - optimal_area), 0.5)
    
    # ── 4. Edge Density Score (NEW) ──────────────────────────────────
    # Well-defined facial features should have high edge density
    # Uses Canny edge detection to measure edge presence
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = np.sum(edges > 0) / (h * w)  # Fraction of pixels with edges
    
    # Good edge density: 5-20% of pixels, avoid noise (>30%) or featureless (<2%)
    if edge_density < 0.02:
        edge_score = 0.2  # Too featureless
    elif edge_density > 0.30:
        edge_score = 0.3  # Too noisy
    elif edge_density < 0.05:
        edge_score = 0.2 + (0.8 * (edge_density - 0.02) / (0.05 - 0.02))
    elif edge_density < 0.20:
        edge_score = 1.0  # Optimal range
    else:
        edge_score = 1.0 - (0.7 * (edge_density - 0.20) / (0.30 - 0.20))
    
    # ── Final score: weighted combination ─────────────────────────────
    # Sharpness: 30%, Contrast+Brightness: 25%, Size: 25%, Edges: 20%
    quality_score = (
        sharpness_score * 0.30 +
        (contrast_score + brightness_score) * 0.25 +
        size_score * 0.25 +
        edge_score * 0.20
    )
    
    quality_score = max(0.0, min(1.0, quality_score))
    
    return round(quality_score, 4)