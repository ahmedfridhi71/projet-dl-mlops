"""Prétraitement des images, identique à src/data/transforms.py (val/test).

Copie volontaire : le backend ne dépend pas du dossier src/.
Un test vérifie que ces constantes restent égales à celles de src.
"""
import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageIllisible(Exception):
    """Les octets reçus ne forment pas une image lisible."""


class FormatNonSupporte(Exception):
    """L'image n'est ni un JPEG ni un PNG."""


def load_image(data: bytes) -> Image.Image:
    """Ouvre les octets reçus et renvoie une image RGB."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ImageIllisible("Image illisible ou corrompue.") from e
    if img.format not in ALLOWED_FORMATS:
        raise FormatNonSupporte(f"Format {img.format} refusé : JPEG ou PNG seulement.")
    return img.convert("RGB")


def resize(img: Image.Image) -> Image.Image:
    """Redimensionne en 224x224 (bilinéaire, comme transforms.Resize)."""
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def to_input(img: Image.Image) -> np.ndarray:
    """Image 224x224 -> tableau float32 [1, 3, 224, 224] normalisé (ToTensor + Normalize)."""
    x = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3], valeurs 0-1
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    x = x.transpose(2, 0, 1)  # [3, H, W]
    return x[np.newaxis].astype(np.float32)  # [1, 3, H, W]
