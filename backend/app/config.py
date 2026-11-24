"""Réglages du backend, lus depuis les variables d'environnement."""
import os
from pathlib import Path

# Racine du dépôt (backend/app/config.py -> on remonte de 2 dossiers).
ROOT = Path(__file__).resolve().parents[2]

MODEL_ONNX_PATH = Path(os.getenv("MODEL_ONNX_PATH", ROOT / "models" / "convnext-tiny.onnx"))
MODEL_PT_PATH = Path(os.getenv("MODEL_PT_PATH", ROOT / "models" / "convnext-tiny.pt"))
# Seuil choisi sur la validation : fissure si P(fissure) >= THRESHOLD.
THRESHOLD = float(os.getenv("THRESHOLD", "0.05"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 Mo
ALLOWED_TYPES = {"image/jpeg", "image/png"}
