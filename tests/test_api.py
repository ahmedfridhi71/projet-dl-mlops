"""Tests de l'API backend (FastAPI).

En CI il n'y a ni FastAPI ni poids du modèle : les tests concernés sont SAUTÉS.
Aucune image du split "test" n'est utilisée : les images sont fabriquées ici.
"""
import ast
import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app import config  # chemins réellement utilisés par l'API

ROOT = Path(__file__).resolve().parents[1]


def image_bytes(fmt="PNG", size=(50, 30)):
    """Petite image aléatoire (taille quelconque) encodée en mémoire."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Client de test : démarre l'app (lifespan) une fois pour tout le module."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # nécessaire à TestClient
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def need_onnx():
    pytest.importorskip("onnxruntime")
    if not config.MODEL_ONNX_PATH.exists():
        pytest.skip("Modèle ONNX absent (normal en CI)")


# --- Constantes de prétraitement --------------------------------------------

def _constantes_src():
    """Lit IMAGENET_MEAN et IMAGENET_STD dans src/data/transforms.py sans l'importer
    (pas besoin de torchvision : le test tourne aussi en CI)."""
    tree = ast.parse((ROOT / "src" / "data" / "transforms.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = node.value
    return ast.literal_eval(consts["IMAGENET_MEAN"]), ast.literal_eval(consts["IMAGENET_STD"])


def test_constantes_pretraitement_identiques_a_src():
    from backend.app import preprocess

    mean, std = _constantes_src()
    assert preprocess.IMAGENET_MEAN == mean
    assert preprocess.IMAGENET_STD == std
    assert preprocess.IMAGE_SIZE == 224  # get_transforms(size=224)


# --- Routes sans modèle -----------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metadata(client):
    r = client.get("/model/metadata")
    assert r.status_code == 200
    m = r.json()
    assert m["format"] == "onnx"
    assert m["input_size"] == [3, 224, 224]
    assert m["classes"] == {"0": "sans fissure", "1": "fissure"}
    assert m["threshold"] == 0.05
    assert set(m["metrics"]) == {"validation", "test"}
    assert m["metrics"]["validation"]["recall"] == pytest.approx(0.867)


def test_415_type_refuse(client):
    r = client.post("/predict", files={"file": ("note.txt", b"bonjour", "text/plain")})
    assert r.status_code == 415


def test_415_gif_deguise_en_png(client):
    gif = image_bytes(fmt="GIF")
    r = client.post("/predict", files={"file": ("faux.png", gif, "image/png")})
    assert r.status_code == 415


def test_413_trop_gros(client):
    gros = b"\xff" * (10 * 1024 * 1024 + 1)
    r = client.post("/predict", files={"file": ("gros.jpg", gros, "image/jpeg")})
    assert r.status_code == 413


def test_400_image_illisible(client):
    r = client.post("/predict", files={"file": ("casse.png", b"pas une image", "image/png")})
    assert r.status_code == 400


# --- Routes avec modèle (sautées si les poids sont absents) -----------------

@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_petite_image(client, need_onnx, fmt, mime):
    r = client.post("/predict", files={"file": ("petite", image_bytes(fmt), mime)})
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["probability"] <= 1.0
    assert d["threshold"] == 0.05
    assert d["inference_ms"] > 0
    attendu = "fissure" if d["probability"] >= d["threshold"] else "sans fissure"
    assert d["label"] == attendu


def test_predict_visualize(client, need_onnx):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not config.MODEL_PT_PATH.exists():
        pytest.skip("Poids PyTorch absents (normal en CI)")
    r = client.post("/predict/visualize", files={"file": ("p.png", image_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    overlay = Image.open(io.BytesIO(base64.b64decode(d["overlay_png_base64"])))
    assert overlay.format == "PNG"
    assert overlay.size == (224, 224)
