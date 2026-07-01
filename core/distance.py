import numpy as np


def calculate_euclidean(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    if vector1.shape != vector2.shape:
        raise ValueError(f"Shape mismatch: {vector1.shape} vs {vector2.shape}")
    
    diff = vector1 - vector2
    return float(np.sqrt(np.sum(np.square(diff))))


def calculate_euclidean_batch(live_vec: np.ndarray, stored_matrix: np.ndarray) -> np.ndarray:
    """Calculate distances from live vector to all stored vectors."""
    diff = stored_matrix - live_vec
    return np.sqrt(np.sum(np.square(diff), axis=1))
