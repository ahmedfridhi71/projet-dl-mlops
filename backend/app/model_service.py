"""Inférence avec ONNX Runtime (CPU)."""
import time
from pathlib import Path

import numpy as np

LABELS = {0: "sans fissure", 1: "fissure"}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Transforme les logits en probabilités (somme = 1)."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class ModelService:
    """Garde la session ONNX en mémoire et fait les prédictions."""

    def __init__(self, onnx_path: Path, threshold: float):
        import onnxruntime as ort  # import ici : le reste de l'API marche sans

        self.threshold = threshold
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name  # "image"

    def predict(self, x: np.ndarray) -> dict:
        """x : tableau [1, 3, 224, 224]. Renvoie label, probabilité, seuil, durée."""
        start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: x})[0]
        inference_ms = (time.perf_counter() - start) * 1000
        proba = float(softmax(logits)[0, 1])  # probabilité de la classe 1 = fissure
        label = LABELS[1] if proba >= self.threshold else LABELS[0]
        return {
            "label": label,
            "probability": proba,
            "threshold": self.threshold,
            "inference_ms": inference_ms,
        }
