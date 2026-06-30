import json

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from core.schemas import (
    AttendanceResponse,
    RegisterEmbeddingResponse,
    DetectionResult,
    ContinuousDetectionResponse,
)
from core.preprocessing import prepare_image, calculate_face_quality
from core.knn import find_match, find_best_student_match, CONTINUOUS_DISTANCE_THRESHOLD
from core.face_detection import detect_faces, crop_face

app = FastAPI(title="Face Recognition ML Service", version="1.1.0")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty. Please upload a valid image.")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    if len(np_arr) == 0:
        raise HTTPException(status_code=400, detail="Could not read image data. File may be corrupted.")

    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. File may not be a valid image format (JPEG, PNG, etc.)",
        )
    return img_bgr


@app.post("/register-embedding", response_model=RegisterEmbeddingResponse)
async def register_embedding(image: UploadFile = File(...)):
    """Extract a HOG embedding from the largest face in an uploaded image."""
    img_bgr = _decode_image(await image.read())

    faces = detect_faces(img_bgr, live=False)
    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please ensure your face is clearly visible.",
        )

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = crop_face(img_bgr, x, y, w, h)
    embedding_vec = prepare_image(face_crop)
    quality_score = calculate_face_quality(face_crop)

    return RegisterEmbeddingResponse(
        embedding=embedding_vec.tolist(),
        quality_score=quality_score,
        status="success",
        message="Face embedding extracted successfully",
    )


@app.post("/process-attendance", response_model=AttendanceResponse)
async def process_attendance(
    image: UploadFile = File(...),
    session_id: str = Form(""),
    stored_vectors: str = Form(""),
    labels: str = Form(""),
):
    """Match the largest face in a still image against stored embeddings."""
    try:
        stored_vecs_list = json.loads(stored_vectors) if stored_vectors else []
        labels_list = json.loads(labels) if labels else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in stored_vectors or labels")

    img_bgr = _decode_image(await image.read())
    faces = detect_faces(img_bgr, live=False)

    if len(faces) == 0:
        return AttendanceResponse(
            student_id=None,
            confidence=0.0,
            distance_to_nearest=float("inf"),
            status="no_face",
        )

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = crop_face(img_bgr, x, y, w, h)
    live_vec = prepare_image(face_crop)

    stored_matrix = np.array(stored_vecs_list, dtype=np.float64)
    result = find_match(live_vec, stored_matrix, labels_list, k=3)
    return AttendanceResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.1.0", "matcher": "per_student_min"}


@app.post("/continuous-detection", response_model=ContinuousDetectionResponse)
async def continuous_detection(
    image: UploadFile = File(...),
    session_id: str = Form(""),
    stored_vectors: str = Form(""),
    labels: str = Form(""),
):
    """
    Detect all faces in a live webcam frame and match each against stored embeddings.
    Uses per-student minimum-distance matching and a slightly relaxed live threshold.
    """
    try:
        stored_vecs_list = json.loads(stored_vectors) if stored_vectors else []
        labels_list = json.loads(labels) if labels else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in stored_vectors or labels")

    img_bgr = _decode_image(await image.read())
    faces = detect_faces(img_bgr, live=True)

    if len(faces) == 0:
        return ContinuousDetectionResponse(
            detections=[],
            total_faces_detected=0,
            status="no_faces",
            nearest_distance=None,
        )

    if not stored_vecs_list or not labels_list:
        return ContinuousDetectionResponse(
            detections=[],
            total_faces_detected=len(faces),
            status="no_stored_vectors",
            nearest_distance=None,
        )

    if len(stored_vecs_list) != len(labels_list):
        raise HTTPException(
            status_code=422,
            detail="stored_vectors and labels must have the same length",
        )

    detections = []
    nearest_distance = None
    stored_matrix = np.array(stored_vecs_list, dtype=np.float64)

    for x, y, w, h in faces:
        face_crop = crop_face(img_bgr, x, y, w, h)
        live_vec = prepare_image(face_crop)

        result = find_best_student_match(live_vec, stored_matrix, labels_list)
        distance = result.get("distance_to_nearest", float("inf"))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance

        if result.get("status") == "identified" and distance <= CONTINUOUS_DISTANCE_THRESHOLD:
            detections.append(
                DetectionResult(
                    student_id=result.get("student_id"),
                    confidence=result.get("confidence", 0.0),
                    distance=distance,
                )
            )

    return ContinuousDetectionResponse(
        detections=detections,
        total_faces_detected=len(faces),
        status="success" if detections else "no_matches",
        nearest_distance=round(nearest_distance, 6) if nearest_distance is not None else None,
    )
