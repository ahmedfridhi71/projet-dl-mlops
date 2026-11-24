"""Point d'entrée de l'API FastAPI."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .gradcam_service import GradCAMService
from .preprocess import FormatNonSupporte, ImageIllisible, load_image, resize, to_input
from .schemas import HealthResponse, MetadataResponse, PredictResponse, VisualizeResponse

log = logging.getLogger("uvicorn.error")
MODEL_CARD = json.loads((Path(__file__).parent / "model_card.json").read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage : on charge le modèle ONNX UNE seule fois.
    app.state.model = None
    if config.MODEL_ONNX_PATH.exists():
        from .model_service import ModelService

        app.state.model = ModelService(config.MODEL_ONNX_PATH, config.THRESHOLD)
        log.info("Modèle ONNX chargé : %s", config.MODEL_ONNX_PATH)
    else:
        log.warning("Modèle ONNX introuvable : %s (/predict renverra 503)", config.MODEL_ONNX_PATH)
    # Service Grad-CAM : les poids PyTorch seront chargés à la première requête /predict/visualize.
    app.state.gradcam = GradCAMService(config.MODEL_PT_PATH)
    yield


app = FastAPI(title="API fissures béton", version="1.0.0", lifespan=lifespan)

# Autorise le frontend Vite (port 5173) à appeler l'API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_upload(file: UploadFile):
    """Vérifie le fichier reçu et renvoie l'image RGB. Lève une erreur HTTP sinon."""
    if file.content_type not in config.ALLOWED_TYPES:
        raise HTTPException(415, "Type de fichier refusé : envoyez une image JPEG ou PNG.")
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Fichier trop gros : 10 Mo maximum.")
    try:
        return load_image(data)
    except FormatNonSupporte as e:
        raise HTTPException(415, str(e))
    except ImageIllisible as e:
        raise HTTPException(400, str(e))


@app.get("/health", response_model=HealthResponse)
def health():
    """Vérifie que le serveur répond."""
    return {"status": "ok"}


@app.get("/model/metadata", response_model=MetadataResponse)
def metadata():
    """Décrit le modèle servi : architecture, classes, seuil, métriques."""
    return {**MODEL_CARD, "threshold": config.THRESHOLD}


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, file: UploadFile = File(...)):
    """Reçoit une image et dit s'il y a une fissure (ONNX Runtime)."""
    img = read_upload(file)
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Modèle non chargé sur le serveur.")
    return model.predict(to_input(resize(img)))


@app.post("/predict/visualize", response_model=VisualizeResponse)
def predict_visualize(request: Request, file: UploadFile = File(...)):
    """Comme /predict, plus l'image avec la carte Grad-CAM de la classe fissure."""
    img = read_upload(file)
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Modèle non chargé sur le serveur.")
    if not config.MODEL_PT_PATH.exists():
        raise HTTPException(503, "Poids PyTorch introuvables : Grad-CAM indisponible.")
    img224 = resize(img)
    x = to_input(img224)
    result = model.predict(x)  # décision : même modèle ONNX que /predict
    result["overlay_png_base64"] = request.app.state.gradcam.overlay(img224, x)
    return result
