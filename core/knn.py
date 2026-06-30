import numpy as np
from collections import Counter
from .distance import calculate_euclidean_batch

# Static single-photo attendance — keep stricter.
DISTANCE_THRESHOLD = 0.55

# Live webcam stream — slightly more tolerant for lighting/angle drift.
CONTINUOUS_DISTANCE_THRESHOLD = 0.68


def find_match(
    live_vec: np.ndarray,
    stored_vecs: np.ndarray,
    labels: list[str],
    k: int = 3,
    threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """
    K-Nearest Neighbours classifier — manual implementation.

    Used for single-frame attendance marking.
    """
    if len(stored_vecs) == 0:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": float("inf"),
            "status": "no_enrolled_students",
        }

    stored_matrix = np.array(stored_vecs)
    label_array = np.array(labels)

    distances = calculate_euclidean_batch(live_vec, stored_matrix)
    sorted_indices = np.argsort(distances)
    k_indices = sorted_indices[:k]
    k_distances = distances[k_indices]
    k_labels = label_array[k_indices].tolist()

    nearest_distance = float(k_distances[0])

    if nearest_distance > threshold:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": nearest_distance,
            "status": "unknown",
        }

    vote_counts = Counter(k_labels)
    winner, votes = vote_counts.most_common(1)[0]
    confidence = votes / k

    return {
        "student_id": winner,
        "confidence": round(confidence, 4),
        "distance_to_nearest": round(nearest_distance, 6),
        "status": "identified",
    }


def find_best_student_match(
    live_vec: np.ndarray,
    stored_vecs: np.ndarray,
    labels: list[str],
    threshold: float = CONTINUOUS_DISTANCE_THRESHOLD,
) -> dict:
    """
    Match a live face against enrolled students.

    Instead of flat k-NN across every stored photo, compute the minimum
    distance to each student's embeddings and pick the closest student.
    This is more reliable when each student has multiple registration photos.
    """
    if len(stored_vecs) == 0:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": float("inf"),
            "status": "no_enrolled_students",
        }

    stored_matrix = np.array(stored_vecs)
    per_student: dict[str, float] = {}

    for student_id in dict.fromkeys(labels):
        indices = [i for i, label in enumerate(labels) if label == student_id]
        student_vecs = stored_matrix[indices]
        per_student[student_id] = float(np.min(
            calculate_euclidean_batch(live_vec, student_vecs)
        ))

    ranked = sorted(per_student.items(), key=lambda item: item[1])
    best_student, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")

    if best_distance > threshold:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": round(best_distance, 6),
            "status": "unknown",
        }

    # Higher confidence when clearly closer than the runner-up and well below threshold.
    margin = second_distance - best_distance if np.isfinite(second_distance) else 0.2
    distance_score = max(0.0, 1.0 - (best_distance / threshold))
    margin_score = min(1.0, margin / 0.15)
    confidence = round(min(0.98, 0.55 + (distance_score * 0.25) + (margin_score * 0.18)), 4)

    return {
        "student_id": best_student,
        "confidence": confidence,
        "distance_to_nearest": round(best_distance, 6),
        "status": "identified",
    }
