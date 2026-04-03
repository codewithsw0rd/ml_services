import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from core.preprocessing import prepare_image, calculate_face_quality
from core.knn import find_match
from core.schemas import AttendanceResponse, RegisterEmbeddingResponse

app = FastAPI(title="Face Recognition ML Service", version="1.0.0")


@app.post("/register-embedding", response_model=RegisterEmbeddingResponse)
async def register_embedding(
    image: UploadFile = File(...),
):
    """
    Face registration endpoint.
    Extracts HOG feature vector from a student's face image.

    Process:
        1. Decode the uploaded image
        2. Detect face using Haar cascade
        3. Crop and normalise the face region
        4. Extract HOG feature vector (1764 dimensions)
        5. Return embedding vector to Django for storage

    Returns:
        RegisterEmbeddingResponse with embedding vector and status
    """
    
    # ── Decode uploaded image ─────────────────────────────────────────
    image_bytes = await image.read()
    np_arr      = np.frombuffer(image_bytes, np.uint8)
    img_bgr     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # ── Face detection (Haar cascade) ──────────────────────────────────
    gray_full  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray_full,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please ensure your face is clearly visible."
        )

    # Use the largest detected face (most prominent in frame)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h  = largest_face
    face_crop    = img_bgr[y : y + h, x : x + w]

    # ── Feature extraction ────────────────────────────────────────────
    embedding_vec = prepare_image(face_crop)     # shape: (1764,)

    # ── Quality scoring ────────────────────────────────────────────────
    quality_score = calculate_face_quality(face_crop)

    # Convert numpy array to list for JSON serialization
    embedding_list = embedding_vec.tolist()

    return RegisterEmbeddingResponse(
        embedding=embedding_list,
        quality_score=quality_score,
        status="success",
        message="Face embedding extracted successfully"
    )


@app.post("/process-attendance", response_model=AttendanceResponse)
async def process_attendance(
    image: UploadFile = File(...),
    session_id: int = 0,
    stored_vectors: str = "",   # JSON-encoded list of lists from Django
    labels: str = "",           # JSON-encoded list of student_id strings
):
    """
    End-to-end attendance pipeline.

    Django sends:
        - image file (multipart)
        - stored_vectors: JSON array of shape (N, D)
        - labels:         JSON array of N student_id strings

    This service:
        1. Decodes and detects faces in the image.
        2. Normalises the detected face region.
        3. Extracts HOG features.
        4. Runs manual KNN against all stored vectors.
        5. Returns the identified student_id (or "unknown").
    """
    import json

    # ── Parse stored data sent by Django ─────────────────────────────
    try:
        stored_vecs_list = json.loads(stored_vectors) if stored_vectors else []
        labels_list      = json.loads(labels)         if labels         else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in stored_vectors or labels")

    # ── Decode uploaded image ─────────────────────────────────────────
    image_bytes = await image.read()
    
    # Check if image data is empty
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty. Please upload a valid image.")
    
    np_arr      = np.frombuffer(image_bytes, np.uint8)
    
    # Check if numpy array is empty
    if len(np_arr) == 0:
        raise HTTPException(status_code=400, detail="Could not read image data. File may be corrupted.")
    
    img_bgr     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may not be a valid image format (JPEG, PNG, etc.)")

    # ── Face detection (Haar cascade — no deep lib required) ──────────
    gray_full  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray_full,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return AttendanceResponse(
            student_id=None,
            confidence=0.0,
            distance_to_nearest=float("inf"),
            status="no_face",
        )

    # Use the largest detected face (most prominent in frame)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h  = largest_face
    face_crop    = img_bgr[y : y + h, x : x + w]

    # ── Feature extraction ────────────────────────────────────────────
    live_vec = prepare_image(face_crop)     # shape: (D,)

    # ── KNN matching ──────────────────────────────────────────────────
    stored_matrix = np.array(stored_vecs_list, dtype=np.float64)
    result        = find_match(live_vec, stored_matrix, labels_list, k=3)

    return AttendanceResponse(**result)


@app.get("/health")
async def health():
    return {"status": "ok"}
