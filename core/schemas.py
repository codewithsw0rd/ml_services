from pydantic import BaseModel
from typing import Optional

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
    