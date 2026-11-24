"""Carte Grad-CAM (PyTorch CPU) : montre où le modèle « voit » la fissure.

ONNX Runtime ne calcule pas de gradients, donc on utilise ici le modèle PyTorch.
Copie autonome de src/models/architecture.py et src/models/gradcam.py.
"""
import base64
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA = 0.45  # transparence de la carte posée sur l'image
CLASSE_FISSURE = 1


def build_convnext_tiny():
    """ConvNeXt-Tiny de torchvision avec une tête à 2 sorties (comme à l'entraînement)."""
    import torch.nn as nn
    from torchvision import models

    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


class GradCAM:
    """Calcule la carte Grad-CAM sur une couche de convolution."""

    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        # On garde les activations et, au backward, leurs gradients.
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=CLASSE_FISSURE):
        """Renvoie une carte 224x224 (numpy, valeurs 0 à 1) pour la classe class_idx."""
        import torch.nn.functional as F

        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)  # importance de chaque canal
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.numpy()


def jet(cam: np.ndarray) -> np.ndarray:
    """Colormap « jet » (bleu = froid, rouge = chaud) : [H, W] 0-1 -> [H, W, 3] 0-1."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_png_base64(img224: Image.Image, cam: np.ndarray, alpha: float = ALPHA) -> str:
    """Pose la carte colorée sur l'image 224x224 et renvoie un PNG encodé en base64."""
    base = np.asarray(img224, dtype=np.float32) / 255.0
    mix = (1 - alpha) * base + alpha * jet(cam)
    out = Image.fromarray((mix * 255).round().astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GradCAMService:
    """Modèle PyTorch chargé à la première demande, puis gardé en mémoire."""

    def __init__(self, pt_path: Path):
        self.pt_path = pt_path
        self.gradcam = None
        self._lock = threading.Lock()  # une seule carte à la fois (hooks partagés)

    def _load(self):
        import torch

        model = build_convnext_tiny()
        model.load_state_dict(torch.load(self.pt_path, map_location="cpu", weights_only=True))
        model.eval()
        self.gradcam = GradCAM(model, model.features[-1])  # dernière couche de convolution

    def overlay(self, img224: Image.Image, x: np.ndarray) -> str:
        """img224 : image redimensionnée ; x : tableau [1, 3, 224, 224] normalisé."""
        import torch

        with self._lock:
            if self.gradcam is None:
                self._load()
            cam = self.gradcam(torch.from_numpy(x))
        return overlay_png_base64(img224, cam)
