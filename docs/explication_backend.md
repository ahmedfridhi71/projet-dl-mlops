# Explication du backend (FastAPI)

Le backend est un petit serveur web. On lui envoie une photo de béton, il répond
« fissure » ou « sans fissure », avec la probabilité. Il peut aussi renvoyer une
image Grad-CAM qui montre où le modèle a regardé.

Modèle servi : **ConvNeXt-Tiny**, au **seuil 0,05** (seuil choisi sur la validation).
Fissure si P(fissure) >= 0,05.

## Les 4 routes

| Route | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Dit si le serveur est vivant : `{"status": "ok"}`. |
| `/model/metadata` | GET | Décrit le modèle : architecture, format, classes, seuil, métriques. |
| `/predict` | POST | Reçoit une image (champ `file`), renvoie label, probabilité, seuil, durée. Utilise ONNX Runtime. |
| `/predict/visualize` | POST | Comme `/predict`, plus l'image avec la carte Grad-CAM en PNG base64. Utilise PyTorch. |

## Rôle de chaque fichier

### `backend/app/__init__.py`
Fichier presque vide qui fait du dossier `app` un paquet Python.
Grâce à lui, les fichiers peuvent s'importer entre eux (`from .preprocess import ...`).

### `backend/app/config.py`
Regroupe tous les réglages à un seul endroit.
Il lit les variables d'environnement `MODEL_ONNX_PATH`, `MODEL_PT_PATH` et `THRESHOLD`
(avec des valeurs par défaut), et fixe les limites : 10 Mo maximum, JPEG ou PNG seulement.

### `backend/app/preprocess.py`
Prépare l'image exactement comme à l'entraînement : RGB, redimensionnement en 224x224,
division par 255, puis normalisation avec la moyenne et l'écart-type d'ImageNet.
Il refuse aussi les fichiers qui ne sont ni JPEG ni PNG, et ceux qui sont illisibles.
C'est une copie de `src/data/transforms.py` pour que le backend soit autonome.

### `backend/app/model_service.py`
Contient la session ONNX Runtime (sur CPU), chargée une seule fois au démarrage.
Il donne l'image au modèle, transforme les 2 scores (logits) en probabilités avec un softmax,
puis compare la probabilité de « fissure » au seuil pour choisir le label.
Il mesure aussi la durée de l'inférence (`inference_ms`).

### `backend/app/gradcam_service.py`
Calcule la carte Grad-CAM avec PyTorch, car ONNX Runtime ne sait pas calculer de gradients.
Le modèle PyTorch n'est chargé qu'à la première demande (il est lent à charger).
La carte (valeurs de 0 à 1) est colorée en « jet » (bleu = froid, rouge = chaud),
posée sur l'image avec une transparence de 0,45, puis renvoyée en PNG base64.

### `backend/app/schemas.py`
Décrit la forme exacte des réponses JSON avec Pydantic (noms des champs et types).
FastAPI s'en sert pour vérifier les réponses et pour générer la documentation `/docs`.

### `backend/app/model_card.json`
La « carte d'identité » du modèle : nom, architecture, format, taille d'entrée, classes,
seuil et métriques de validation et de test. Ces métriques viennent de
`reports/metrics/convnext-tiny-r85_val.json` et `convnext-tiny-final_test.json`.
La route `/model/metadata` renvoie ce fichier.

### `backend/app/main.py`
Le cœur de l'API : crée l'application FastAPI, active le CORS pour le frontend
(port 5173) et définit les 4 routes. Au démarrage (lifespan), il charge le modèle ONNX une seule fois.
Il vérifie le fichier reçu et renvoie l'erreur adaptée : 415 si le type est mauvais,
413 si le fichier est trop gros, 400 si l'image est illisible, 503 si le modèle manque.

### `backend/requirements.txt`
La liste des bibliothèques Python nécessaires au backend.
Dans Docker, torch sera installé en version CPU pour que l'image soit plus légère.
`httpx` n'est là que pour les tests (TestClient en a besoin).

### `tests/test_api.py`
Les tests automatiques de l'API, lancés par `pytest -q`.
Ils vérifient les routes, les erreurs 415, 413 et 400, et que les constantes de prétraitement
sont les mêmes que dans `src`. Sans fastapi ou sans fichiers modèle (comme en CI),
les tests concernés sont sautés au lieu d'échouer.

### `docs/explication_backend.md`
Ce document.

> Les fichiers vides `backend/app/inference.py`, `backend/tests/test_api.py` et
> `backend/Dockerfile` viennent d'un ancien squelette. Ils ne sont utilisés par rien
> et n'ont pas été modifiés.

## Le chemin d'une requête `/predict`, de bout en bout

1. **Le client envoie l'image.** Par exemple :
   `curl.exe -F "file=@photo.jpg" http://127.0.0.1:8000/predict`.
   L'image part en « multipart », dans le champ `file`.
2. **FastAPI reçoit la requête** et appelle la fonction `predict` de `main.py`.
   Le modèle ONNX est déjà en mémoire : il a été chargé au démarrage du serveur.
3. **Vérification du fichier** (`read_upload` dans `main.py`) :
   - le type n'est pas `image/jpeg` ou `image/png` : **415** ;
   - le fichier dépasse 10 Mo : **413** ;
   - PIL n'arrive pas à ouvrir l'image : **400** ;
   - l'image s'ouvre mais n'est ni JPEG ni PNG (un GIF renommé, par exemple) : **415**.
4. **Prétraitement** (`preprocess.py`) : l'image est convertie en RGB, redimensionnée
   en 224x224, puis transformée en tableau de nombres `[1, 3, 224, 224]` normalisé.
5. **Inférence** (`model_service.py`) : ONNX Runtime calcule 2 scores (logits),
   un pour « sans fissure » et un pour « fissure ».
6. **Softmax** : les 2 scores deviennent 2 probabilités dont la somme vaut 1.
   On garde celle de la classe 1, « fissure ».
7. **Décision** : si cette probabilité est >= 0,05, le label est « fissure »,
   sinon « sans fissure ».
8. **Réponse JSON** (forme définie dans `schemas.py`) :
   ```json
   {"label": "fissure", "probability": 0.153, "threshold": 0.05, "inference_ms": 31.7}
   ```

Pour `/predict/visualize`, les étapes 1 à 7 sont les mêmes. Ensuite, `gradcam_service.py`
calcule la carte Grad-CAM pour la classe fissure avec PyTorch, la superpose à l'image
224x224 et ajoute `overlay_png_base64` à la réponse.

## Pourquoi ces choix (questions possibles à l'oral)

- **Pourquoi un seuil de 0,05 et pas 0,5 ?** Rater une fissure est plus grave qu'une fausse alerte.
  Un seuil bas augmente le rappel. Il a été choisi sur la validation, jamais sur le test.
- **Pourquoi ONNX pour `/predict` ?** ONNX Runtime est léger et rapide sur CPU
  (environ 30 à 45 ms par image). On a vérifié qu'il donne la même probabilité que PyTorch
  (écart d'environ 1e-7).
- **Pourquoi PyTorch pour Grad-CAM ?** Grad-CAM a besoin des gradients, et ONNX Runtime ne les calcule pas.
- **Pourquoi copier le code au lieu d'importer `src` ?** Pour que le backend soit autonome
  (plus simple à mettre dans Docker). Un test vérifie que les constantes restent identiques à `src`.
- **Pourquoi charger le modèle au démarrage ?** Le charger à chaque requête serait très lent.
  On le charge une fois, puis on le réutilise.

## Lancer le backend

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs
