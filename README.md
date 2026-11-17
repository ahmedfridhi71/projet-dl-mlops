# Détection de fissures du béton (SDNET2018)

[![CI tests](https://github.com/ahmedfridhi71/projet-dl-mlops/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ahmedfridhi71/projet-dl-mlops/actions/workflows/ci-tests.yml)

Projet du module **Deep Learning Avancé** (TEK-UP) : classification d'images de béton
(fissure / sans fissure) par transfer learning, avec un cycle de vie MLOps complet
(DVC, MLflow, ONNX, Docker, GitHub Actions) et une application FastAPI + React.

**Axe retenu :** CNN avancé et Transfer Learning, avec explicabilité (Grad-CAM).

## Dataset

**SDNET2018** : 56 092 patchs JPG de 256x256 pixels, issus de 230 photos de ponts (D),
chaussées (P) et murs (W). Étiquette binaire donnée par le dossier : `C` (fissure) ou
`U` (sans fissure). Environ 15,1 % de fissures.

- Source : [https://digitalcommons.usu.edu/all_datasets/48](https://digitalcommons.usu.edu/all_datasets/48)
- DOI : [https://doi.org/10.15142/T3TD19](https://doi.org/10.15142/T3TD19)
- Licence : CC BY 4.0

Le dataset ne fournit pas de découpage. Les patchs d'une même photo étant très
corrélés, le découpage train / val / test (70 / 15 / 15, graine 42) se fait **par photo
d'origine**, séparément pour chaque surface.

## Structure du projet

```
.github/workflows/    CI (tests) et build Docker (à compléter)
data/                 données, versionnées par DVC (non présentes sur GitHub)
models/               poids des modèles, versionnés par DVC (fichiers .dvc dans Git)
notebooks/            exploration du dataset
reports/              figures et rapport de qualité des données
src/data/             index et découpage, contrôle qualité, Dataset, transforms, loaders
src/models/           ResNet-18, entraînement, évaluation, analyse du seuil
src/utils/            métriques de classification
tests/                tests automatiques (pytest)
backend/              API FastAPI (à venir)
frontend/             interface React (à venir)
pytest.ini            configuration des tests
requirements.txt      dépendances : données, tests, DVC (utilisé par la CI)
requirements-gpu.txt  PyTorch avec CUDA 12.8 et MLflow (entraînement)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pour l'entraînement sur GPU (PyTorch avec CUDA 12.8 et MLflow) :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Environnement testé : Python 3.11, PyTorch 2.11.0, GPU NVIDIA RTX 4050 Laptop (6 Go).

## Récupérer les données

Les images ne sont pas dans Git : elles sont suivies par DVC (`data/raw/SDNET2018_images.dvc`
et `data/processed.dvc`).

```powershell
.\.venv\Scripts\dvc.exe pull
```

> L'entrepôt DVC est pour l'instant **local** (`C:\Users\ahmed\dvc-storage`). Un
> entrepôt partagé sera configuré en Phase 5. En attendant, le dataset peut être
> téléchargé depuis le lien ci-dessus puis placé dans `data/raw/SDNET2018_images`.

## Commandes utiles

```powershell
# Regénérer l'index et le découpage par photo
.\.venv\Scripts\python.exe src\data\make_index.py

# Marquer les doublons à exclure (à relancer après make_index.py)
.\.venv\Scripts\python.exe src\data\flag_exclusions.py

# Contrôle qualité (images corrompues, doublons exacts, planche d'exemples)
.\.venv\Scripts\python.exe src\data\check_quality.py

# Entraîner la baseline (suivi MLflow automatique)
.\.venv\Scripts\python.exe -m src.models.train --epochs 5 --run-name resnet18-baseline

# Analyser le seuil de décision d'un modèle sauvegardé
.\.venv\Scripts\python.exe -m src.models.threshold_analysis --weights models/baseline.pt

# Ouvrir l'interface MLflow (http://127.0.0.1:5000)
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db

# Récupérer le modèle baseline versionné avec DVC
.\.venv\Scripts\dvc.exe pull models\baseline.pt.dvc

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Qualité des données (résultats du contrôle)

- 0 image corrompue ; toutes les images font 256x256 en RGB.
- 13 groupes de doublons exacts (0,02 %), dont 2 avec des étiquettes contradictoires ;
  15 images sont exclues (colonne `exclure` de l'index, aucune dans le test).
- Aucun doublon réparti sur plusieurs splits (aucune fuite).

## Entraînement et suivi des essais

- Baseline : ResNet-18 pré-entraîné sur ImageNet, nouvelle tête à 2 sorties, Adam (lr 1e-4),
  précision mixte FP16, 5 époques (environ 4 minutes par époque sur GPU).
- Résultat sur la validation (seuil 0,5) : PR-AUC 0,882, recall 0,71, précision 0,89.
- Modèle versionné avec DVC : le pointeur `models/baseline.pt.dvc` est dans Git,
  le fichier de poids est dans le remote DVC.
- Chaque essai est suivi par MLflow (expérience `baseline-sdnet`) : réglages, courbes par
  époque, modèle et commit du code.
- Une loss pondérée (poids 3 sur la classe fissure) a été testée : elle déplace le seuil de
  décision sans améliorer la PR-AUC. Le seuil sera choisi sur la validation à la fin.

## Avancement

| Phase | Contenu                                                      | État                            |
| ----- | ------------------------------------------------------------ | -------------------------------- |
| 1     | Données, index, exploration, DVC, CI minimal                | Fait                             |
| 2     | Chargement des données, métriques, baseline ResNet, MLflow | Fait (baseline PR-AUC val 0,882) |
| 3     | Modèles avancés, augmentation, Grad-CAM                    | À venir                         |
| 4     | Export ONNX, API FastAPI, interface React                    | À venir                         |
| 5     | Docker, docker-compose, CI/CD complet                        | À venir                         |
| 6     | Rapport et soutenance                                        | À venir                         |

## Référence

Maguire, M., Dorafshan, S., Thomas, R. J. (2018). *SDNET2018: A concrete crack image
dataset for machine learning applications*. Utah State University.
