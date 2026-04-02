import numpy as np

def calculate_euclidean(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Euclidean distance between two 1-D vectors using only NumPy.

    Formula:  d = √ Σ (v1ᵢ − v2ᵢ)²

    The subtraction, squaring, summation, and square-root are all
    explicit NumPy primitives — no scipy, no sklearn, nothing else.

    Args:
        vector1: 1-D float numpy array, shape (D,)
        vector2: 1-D float numpy array, shape (D,)

    Returns:
        Scalar float — the straight-line distance in D-dimensional space.

    Raises:
        ValueError if the vectors have different shapes.
    """
    
    if vector1.shape != vector2.shape:
        raise ValueError(
            f"Shape mismatch: {vector1.shape} vs {vector2.shape}."
            "Both vectors must have the same dimentionality."
        )
        
    diff = vector1-vector2
    squared = np.square(diff)
    sum_squares = np.sum(squared)
    distance = np.sqrt(sum_squares)
    
    return float(distance)

def calculate_euclidean_batch(
    live_vec: np.ndarray,
    stored_matrix: np.ndarray
) -> np.ndarray:
    """
    Vectorised version: compute distance from live_vec to every row
    in stored_matrix in one NumPy call — no Python loop needed.

    This is what find_match() actually calls internally; the scalar
    version above exists for clarity and unit-testing.

    Args:
        live_vec:      shape (D,)
        stored_matrix: shape (N, D)

    Returns:
        distances: shape (N,)  — one distance per stored vector
    """
    diff = stored_matrix - live_vec
    distances = np.sqrt(np.sum(np.square(diff), axis=1))
    return distances