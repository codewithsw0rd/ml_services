import numpy as np
from collections import Counter
from .distance import calculate_euclidean_batch

DISTANCE_THRESHOLD = 0.55

def find_match(
    live_vec: np.ndarray,
    stored_vecs: np.ndarray,
    labels: list[str],
    k: int = 3,
) -> dict:
    """
    K-Nearest Neighbours classifier — manual implementation.

    Algorithm:
        1. Compute Euclidean distance from live_vec to every stored vector.
        2. Argsort to find indices of k smallest distances.
        3. Collect the k corresponding labels.
        4. Majority vote: the label with the highest count wins.
        5. Confidence = winning_votes / k  (e.g. 2/3 ≈ 0.67).
        6. If the nearest neighbour exceeds DISTANCE_THRESHOLD,
           return "unknown" — the face is too far from all known students.

    Args:
        live_vec:    HOG feature vector of the incoming face, shape (D,)
        stored_vecs: Matrix of all enrolled face vectors,   shape (N, D)
        labels:      Student-ID strings, len N, parallel to stored_vecs
        k:           Number of neighbours to consider (default 3)

    Returns:
        dict with keys: student_id, confidence, distance_to_nearest, status
    """
    if len(stored_vecs) == 0:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": float("inf"),
            "status": "no_enrolled_students",
        }

    stored_matrix = np.array(stored_vecs)        # (N, D)
    label_array   = np.array(labels)             # (N,)

    # ── Step 1: All distances in one vectorised call ─────────────────
    distances = calculate_euclidean_batch(live_vec, stored_matrix)

    # ── Step 2: Argsort → k nearest indices ──────────────────────────
    # np.argsort returns indices that would sort the array ascending.
    # We slice the first k to get the k smallest distances.
    sorted_indices  = np.argsort(distances)          # full sort, O(N log N)
    k_indices       = sorted_indices[:k]             # top-k closest
    k_distances     = distances[k_indices]
    k_labels        = label_array[k_indices].tolist()

    nearest_distance = float(k_distances[0])

    # ── Step 3: Guard — reject if too far from all stored faces ──────
    if nearest_distance > DISTANCE_THRESHOLD:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": nearest_distance,
            "status": "unknown",
        }

    # ── Step 4: Majority vote ─────────────────────────────────────────
    # Counter gives us {label: vote_count, ...}
    # most_common(1) returns [(label, count)] for the winner.
    vote_counts     = Counter(k_labels)
    winner, votes   = vote_counts.most_common(1)[0]
    confidence      = votes / k

    return {
        "student_id": winner,
        "confidence": round(confidence, 4),
        "distance_to_nearest": round(nearest_distance, 6),
        "status": "identified",
    }