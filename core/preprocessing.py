import cv2
import numpy as np
from skimage.feature import hog

TARGET_SIZE = (100, 100)
HOG_ORIENTATION = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def prepare_image(img: np.ndarray) -> np.ndarray:
    """Extract HOG features from face crop and normalize."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = clahe.apply(resized)
    denoised = cv2.bilateralFilter(equalised, d=5, sigmaColor=50, sigmaSpace=50)
    
    feature_vec = hog(
        denoised,
        orientations=HOG_ORIENTATION,
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


def prepare_image_flip_pair(img: np.ndarray) -> list[np.ndarray]:
    """Generate embeddings for original and horizontally flipped image."""
    return [prepare_image(img), prepare_image(cv2.flip(img, 1))]


def calculate_face_quality(face_crop: np.ndarray) -> float:
    """Calculate face quality score (0-1) based on sharpness, contrast, brightness, and size."""
    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop.copy()
    
    h, w = gray.shape
    
    # Sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 2500.0, 1.0)
    
    # Contrast and brightness
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    contrast_score = (std_intensity / 128.0) * 0.5
    brightness_score = max((1.0 - abs(mean_intensity - 128.0) / 128.0) * 0.5, 0.0)
    
    # Face size
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
    
    # Weighted combination
    quality_score = (
        sharpness_score * 0.50 +
        (contrast_score + brightness_score) * 0.30 +
        size_score * 0.20
    )
    
    return round(max(0.0, min(1.0, quality_score)), 4)
