"""Format des réponses JSON (schémas Pydantic)."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    label: str  # "fissure" ou "sans fissure"
    probability: float  # probabilité de la classe fissure
    threshold: float
    inference_ms: float


class VisualizeResponse(PredictResponse):
    overlay_png_base64: str  # image 224x224 + carte Grad-CAM, en PNG base64


class MetadataResponse(BaseModel):
    model: str
    architecture: str
    format: str
    input_size: list[int]
    classes: dict[str, str]
    threshold: float
    metrics: dict[str, dict[str, float]]
