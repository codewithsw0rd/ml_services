from pydantic import BaseModel
from typing import Optional, List

class AttendanceRequest(BaseModel):
    """
        Django sends stored embeddings alongside the live image so the ML services stays stateless - no DB connection needed here.
    """
    session_id : str
    stored_vectors : list[list[float]]
    labels: list[str]
    
class AttendanceResponse(BaseModel):
    student_id : Optional[str]
    confidence : float
    distance_to_nearest : float
    status : str


class RegisterEmbeddingResponse(BaseModel):
    """
    Response for face registration endpoint.
    Contains the extracted embedding vector (1764 dimensions from HOG)
    and quality score of the detected face.
    """
    embedding : list[float]
    quality_score : float
    status : str
    message : str
    

class FaceBox(BaseModel):
    """Bounding box for a detected face (in original image coordinates)."""
    x: int
    y: int
    w: int
    h: int
    status: str  # "identified" | "unknown" | "ambiguous"
    student_id: Optional[str] = None
    confidence: float = 0.0


class DetectionResult(BaseModel):
    """Single face detection result"""
    student_id: Optional[str] = None
    confidence: float
    distance: float
    bbox: Optional[FaceBox] = None


class ContinuousDetectionResponse(BaseModel):
    """Response for continuous face detection endpoint.
    Returns list of all detected faces with their matches."""
    detections: List[DetectionResult]
    total_faces_detected: int
    status: str  # "success", "no_faces", "no_matches"
    nearest_distance: Optional[float] = None
    faces: List[FaceBox] = []  # all detected face bboxes with their match status