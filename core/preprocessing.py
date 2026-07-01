import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FACENET_MODEL = InceptionResnetV1(pretrained='vggface2').to(DEVICE).eval()


def prepare_image(img: np.ndarray) -> np.ndarray:
    """Extract FaceNet embedding from face crop and normalize."""
    resized = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LANCZOS4)
    
    img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    
    with torch.no_grad():
        embedding = FACENET_MODEL(img_tensor.to(DEVICE))
    
    embedding_np = embedding.squeeze().cpu().numpy()
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm
    
    return embedding_np


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
