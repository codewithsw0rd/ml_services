import json
import os
from dotenv import load_dotenv

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from core.preprocessing import prepare_image, calculate_face_quality, prepare_image_flip_pair
from core.knn import find_match, find_best_student_match, CONTINUOUS_DISTANCE_THRESHOLD
from core.face_detection import detect_faces, extract_largest_face, crop_face
from core.schemas import (
    AttendanceResponse,
    RegisterEmbeddingResponse,
    DetectionResult,
    ContinuousDetectionResponse,
    FaceBox,
)

load_dotenv()

app = FastAPI(title="Face Recognition ML Service", version="1.2.1")


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
    """Extract FaceNet embedding from largest face in image."""
    img_bgr = _decode_image(await image.read())
    face_crop = extract_largest_face(img_bgr)

    if face_crop is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please ensure your face is clearly visible.",
        )

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
    face_crop = extract_largest_face(img_bgr)

    if face_crop is None:
        return AttendanceResponse(
            student_id=None,
            confidence=0.0,
            distance_to_nearest=float("inf"),
            status="no_face",
        )

    live_vec = prepare_image(face_crop)
    stored_matrix = np.array(stored_vecs_list, dtype=np.float64)
    result = find_match(live_vec, stored_matrix, labels_list, k=3)
    return AttendanceResponse(**result)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.2.2",
        "matcher": "per_student_min_flip",
        "frame_width": 640,
        "threshold": CONTINUOUS_DISTANCE_THRESHOLD,
    }


@app.post("/continuous-detection", response_model=ContinuousDetectionResponse)
async def continuous_detection(
    image: UploadFile = File(...),
    session_id: str = Form(""),
    stored_vectors: str = Form(""),
    labels: str = Form(""),
):
    """Detect all faces in a live frame and match each against stored embeddings."""
    try:
        stored_vecs_list = json.loads(stored_vectors) if stored_vectors else []
        labels_list = json.loads(labels) if labels else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in stored_vectors or labels")

    img_bgr = _decode_image(await image.read())
    faces = detect_faces(img_bgr)

    if len(faces) == 0:
        return ContinuousDetectionResponse(
            detections=[],
            total_faces_detected=0,
            status="no_faces",
            nearest_distance=None,
            faces=[],
        )

    if not stored_vecs_list or not labels_list:
        # Return bboxes without matches so we can still draw red boxes
        raw_boxes = [
            FaceBox(x=int(x), y=int(y), w=int(w), h=int(h),
                    status="unknown", student_id=None, confidence=0.0)
            for (x, y, w, h) in faces
        ]
        return ContinuousDetectionResponse(
            detections=[],
            total_faces_detected=len(faces),
            status="no_stored_vectors",
            nearest_distance=None,
            faces=raw_boxes,
        )

    if len(stored_vecs_list) != len(labels_list):
        raise HTTPException(
            status_code=422,
            detail="stored_vectors and labels must have the same length",
        )

    stored_matrix = np.array(stored_vecs_list, dtype=np.float64)

    # ── Per-face processing ────────────────────────────────────────────────
    detections: list[DetectionResult] = []
    face_boxes: list[FaceBox] = []
    overall_nearest = float("inf")

    for (x, y, w, h) in faces:
        face_crop = crop_face(img_bgr, x, y, w, h)
        if face_crop is None or face_crop.size == 0:
            face_boxes.append(
                FaceBox(x=int(x), y=int(y), w=int(w), h=int(h),
                        status="unknown", student_id=None, confidence=0.0)
            )
            continue

        live_vecs = prepare_image_flip_pair(face_crop)
        result = find_best_student_match(live_vecs, stored_matrix, labels_list)
        distance = result.get("distance_to_nearest", float("inf"))
        status = result.get("status", "unknown")  # "identified"|"unknown"|"ambiguous"
        student_id = result.get("student_id")
        confidence = result.get("confidence", 0.0)

        overall_nearest = min(overall_nearest, distance)

        face_box = FaceBox(
            x=int(x), y=int(y), w=int(w), h=int(h),
            status=status,
            student_id=student_id,
            confidence=round(confidence, 4),
        )
        face_boxes.append(face_box)

        if status == "identified" and distance <= CONTINUOUS_DISTANCE_THRESHOLD:
            detections.append(
                DetectionResult(
                    student_id=student_id,
                    confidence=confidence,
                    distance=distance,
                    bbox=face_box,
                )
            )

    return ContinuousDetectionResponse(
        detections=detections,
        total_faces_detected=len(faces),
        status="success" if detections else "no_matches",
        nearest_distance=round(overall_nearest, 6) if overall_nearest != float("inf") else None,
        faces=face_boxes,
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    host = "0.0.0.0"
    
    print(f"\n{'='*60}")
    print(f"Starting ML Service")
    print(f"{'='*60}")
    print(f"Host: {host}:{port}")
    print(f"URL: http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"API Docs: http://localhost:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=host, port=port)
