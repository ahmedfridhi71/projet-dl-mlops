# Image du backend FastAPI (detection de fissures, ConvNeXt-Tiny), en deux etapes.
#
# Construction (depuis la racine du depot) :
#   docker build -t fissures-backend backend
# Lancement : les modeles ne sont PAS dans l'image (ni dans Git), on monte le dossier models/ :
#   docker run -p 8000:8000 -v ${PWD}/models:/models:ro fissures-backend

# ---------- Etape 1 : installer les dependances dans un environnement isole ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# torch et torchvision en version CPU (bien plus legere que la version CUDA).
# Memes versions que pour l'entrainement.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0

# Les autres dependances (torch et torchvision sont deja installes).
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Etape 2 : image finale, legere ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# On recupere seulement l'environnement construit a l'etape 1.
COPY --from=builder /opt/venv /opt/venv

# Le code de l'API (seulement le dossier app/).
COPY app/ ./app/

# Chemins des modeles dans le conteneur (dossier monte avec -v) et seuil de decision.
ENV MODEL_ONNX_PATH=/models/convnext-tiny.onnx \
    MODEL_PT_PATH=/models/convnext-tiny.pt \
    THRESHOLD=0.05

# Utilisateur sans droits administrateur, par securite.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker verifie regulierement que l'API repond.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0 : accepte les connexions venant de l'exterieur du conteneur.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]