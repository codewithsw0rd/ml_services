import numpy as np
from collections import Counter
from .distance import calculate_euclidean_batch

# Static single-photo attendance — keep stricter.
DISTANCE_THRESHOLD = 0.55

# Live webcam stream — tolerant for lighting/angle/compression drift.
# HOG + Edge Euclidean distances on compressed webcam JPEGs naturally sit
# higher than clean registration crops; empirically 0.72-0.78 is the
# "close but no match" zone, so we push the ceiling to 0.82 to pass it.
CONTINUOUS_DISTANCE_THRESHOLD = 0.82

# When multiple students are enrolled, require this gap between 1st and 2nd best.
MIN_STUDENT_MARGIN = 0.06


def find_match(
    live_vec: np.ndarray,
    stored_vecs: np.ndarray,
    labels: list[str],
    k: int = 3,
    threshold: float = DISTANCE_THRESHOLD,
) -> dict:
    """
    K-Nearest Neighbours (KNN) classifier — manual implementation.
    
    Algorithm from Computer Science Syllabus: Classification & Pattern Matching
    
    KNN works by:
    1. Computing distance from live face to all stored faces
    2. Finding k nearest neighbors (k=3 here)
    3. Taking majority vote among those k neighbors
    4. Confidence = (votes / k)

    Used for single-frame attendance marking (stricter threshold).
    
    Args:
        live_vec: 1-D embedding from detected face
        stored_vecs: List of 1-D embeddings from enrolled students
        labels: List of student IDs corresponding to stored vectors
        k: Number of neighbors to consider
        threshold: Distance threshold for matching (0.55 for strict)
    
    Returns:
        dict with: student_id, confidence, distance_to_nearest, status
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

    # ── 1. Compute distances to all stored vectors ───────────────────
    distances = calculate_euclidean_batch(live_vec, stored_matrix)
    
    # ── 2. Find k nearest neighbors ──────────────────────────────────
    sorted_indices = np.argsort(distances)
    k_indices = sorted_indices[:k]
    k_distances = distances[k_indices]
    k_labels = label_array[k_indices].tolist()

    nearest_distance = float(k_distances[0])

    # ── 3. Check threshold ───────────────────────────────────────────
    if nearest_distance > threshold:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": nearest_distance,
            "status": "unknown",
        }

    # ── 4. Majority voting among k neighbors ─────────────────────────
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
    live_vec,
    stored_vecs: np.ndarray,
    labels: list[str],
    threshold: float = CONTINUOUS_DISTANCE_THRESHOLD,
) -> dict:
    """
    Match a live face against enrolled students using per-student minimum distance.
    
    Algorithm: Per-Student Minimum Distance Matching
    
    For continuous/live detection, use a different strategy:
    1. For EACH student: find their closest stored embedding
    2. Track per-student distances
    3. Return the student with minimum overall distance
    4. Check gap between best and second-best to detect ambiguity
    
    This handles multiple enrollments per student better than KNN voting.
    
    Used for live webcam stream with flip-pair augmentation (more tolerant).

    Accepts one embedding vector or a list of variants (e.g. original + flipped).
    Uses the minimum distance to each student's stored photos.
    """
    if len(stored_vecs) == 0:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": float("inf"),
            "status": "no_enrolled_students",
        }

    if isinstance(live_vec, list):
        live_vectors = live_vec
    else:
        live_vectors = [live_vec]

    stored_matrix = np.array(stored_vecs)
    per_student: dict[str, float] = {}

    # ── 1. For each student, find best match ────────────────────────
    for student_id in dict.fromkeys(labels):
        # Get all stored embeddings for this student
        indices = [i for i, label in enumerate(labels) if label == student_id]
        student_vecs = stored_matrix[indices]
        best_for_student = float("inf")
        
        # Try each live vector variant (original + flipped)
        for vector in live_vectors:
            best_for_student = min(
                best_for_student,
                float(np.min(calculate_euclidean_batch(vector, student_vecs))),
            )
        per_student[student_id] = best_for_student

    # ── 2. Rank students by distance ────────────────────────────────
    ranked = sorted(per_student.items(), key=lambda item: item[1])
    best_student, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")

    # ── 3. Check threshold ──────────────────────────────────────────
    if best_distance > threshold:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": round(best_distance, 6),
            "status": "unknown",
        }

    # ── 4. Check for ambiguity (too close between 1st and 2nd) ──────
    if len(ranked) > 1 and (second_distance - best_distance) < MIN_STUDENT_MARGIN:
        return {
            "student_id": None,
            "confidence": 0.0,
            "distance_to_nearest": round(best_distance, 6),
            "status": "ambiguous",
        }

    # ── 5. Compute confidence score ─────────────────────────────────
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