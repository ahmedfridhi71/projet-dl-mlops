# Image du backend FastAPI (détection de fissures, ConvNeXt-Tiny).
#
# Construction (depuis la racine du dépôt) :
#   docker build -t fissures-backend backend
# Lancement : les modèles ne sont PAS dans l'image (ni dans Git), on monte le dossier models/ :
#   docker run -p 8000:8000 -v ${PWD}/models:/models:ro fissures-backend

FROM python:3.11-slim

# Pas de fichiers .pyc, logs affichés tout de suite, pas de cache pip dans l'image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) torch et torchvision en version CPU (bien plus légère que la version CUDA).
#    Mêmes versions que pour l'entraînement.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0

# 2) Les autres dépendances. torch et torchvision sont déjà installés : pip ne les retélécharge pas.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Le code de l'API (seulement le dossier app/).
COPY app/ ./app/

# Chemins des modèles dans le conteneur (dossier monté avec -v) et seuil de décision.
ENV MODEL_ONNX_PATH=/models/convnext-tiny.onnx \
    MODEL_PT_PATH=/models/convnext-tiny.pt \
    THRESHOLD=0.05

# Utilisateur sans droits administrateur, par sécurité.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker vérifie régulièrement que l'API répond.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0 : accepte les connexions venant de l'extérieur du conteneur.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
