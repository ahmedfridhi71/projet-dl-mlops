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
docs/                 pitch de mi-parcours (PDF)
reports/              figures, rapport de qualité des données, résultats (reports/metrics)
src/data/             index et découpage, contrôle qualité, Dataset, transforms, loaders
src/models/           ResNet-18, EfficientNet-V2-S, ConvNeXt-Tiny, entraînement, seuils, Grad-CAM, export ONNX
src/utils/            métriques de classification
tests/                tests automatiques (pytest)
backend/              API FastAPI (/health, /model/metadata, /predict, /predict/visualize), Dockerfile
frontend/             interface React (upload, résultat, superposition Grad-CAM), Dockerfile nginx
docker-compose.yml    lance le backend (port 8000) et le frontend (port 80)
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

# Entraîner un autre modèle (resnet18, efficientnet_v2_s, convnext_tiny)
.\.venv\Scripts\python.exe -m src.models.train --model efficientnet_v2_s --run-name efficientnet-v2-s --num-workers 4

# Mesurer la vitesse et la mémoire GPU des modèles
.\.venv\Scripts\python.exe -m src.models.bench 32

# Analyser le seuil de décision d'un modèle sauvegardé
.\.venv\Scripts\python.exe -m src.models.threshold_analysis --weights models/efficientnet-v2-s.pt --model efficientnet_v2_s --name efficientnet-v2-s

# Mesurer un modèle par surface à un seuil donné (JSON)
.\.venv\Scripts\python.exe -m src.models.export_metrics --weights models/efficientnet-v2-s.pt --model efficientnet_v2_s --name efficientnet-v2-s-r80 --split val --threshold 0.4

# Tableaux de comparaison (CSV)
.\.venv\Scripts\python.exe -m src.models.compare_archs
.\.venv\Scripts\python.exe -m src.models.compare_surfaces

# Grad-CAM : où le modèle regarde (figures dans reports/figures)
.\.venv\Scripts\python.exe -m src.models.gradcam --weights models/convnext-tiny.pt --model convnext_tiny --name convnext-tiny --threshold 0.05

# Exporter le modèle final en ONNX (models/convnext-tiny.onnx)
.\.venv\Scripts\python.exe -m src.models.export_onnx

# Ouvrir l'interface MLflow (http://127.0.0.1:5000)
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db

# Enregistrer les mesures finales dans MLflow (expérience fissures-beton-sdnet2018)
.\.venv\Scripts\python.exe src\models\log_to_mlflow.py

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

Tous les modèles sont entraînés avec le même `train.py` (Adam lr 1e-4, batch 32, FP16,
5 époques) : seule l'option `--model` change. La meilleure époque est choisie par la
PR-AUC de validation. Chaque essai est suivi par MLflow (expérience `baseline-sdnet`) et
écrit son historique dans `reports/metrics/<essai>_epochs.csv`. Les mesures finales
(fichiers JSON de `reports/metrics/`) sont rangées dans l'expérience
`fissures-beton-sdnet2018`.

### Phase 2 : baseline et déséquilibre de classes

- Baseline : ResNet-18 pré-entraîné sur ImageNet, nouvelle tête à 2 sorties.
- Modèle versionné avec DVC : le pointeur `models/baseline.pt.dvc` est dans Git, le
  fichier de poids est dans le remote DVC.
- Cinq solutions au déséquilibre (15 % de fissures) : mesure adaptée, loss pondérée
  (poids 3), échantillonnage équilibré, augmentation, choix du seuil. Les trois
  premières ne font que déplacer le point recall / précision : la PR-AUC reste proche
  de 0,88 (baseline 0,882, poids 3 : 0,880, équilibré : 0,879).

### Phase 3.1 : comparaison de trois architectures

Validation, meilleure époque, seuil 0,5 :

| Modèle           | Époque | PR-AUC | Recall | Précision | F1    |
| ----------------- | ------- | ------ | ------ | ---------- | ----- |
| ResNet-18         | 3       | 0,882  | 0,713  | 0,892      | 0,792 |
| EfficientNet-V2-S | 5       | 0,893  | 0,779  | 0,864      | 0,819 |
| ConvNeXt-Tiny     | 2       | 0,900  | 0,597  | 0,981      | 0,742 |

Comme les recalls diffèrent au seuil 0,5, la comparaison juste se fait **à recall égal**
(environ 0,80), avec un seuil par modèle (recall / précision) :

| Surface | ResNet-18 (seuil 0,3) | EfficientNet-V2-S (0,4) | ConvNeXt-Tiny (0,1) |
| ------- | --------------------- | ----------------------- | ------------------- |
| Global  | 0,794 / 0,811         | 0,809 / 0,822           | 0,803 / 0,849       |
| Ponts   | 0,682 / 0,762         | 0,729 / 0,839           | 0,710 / 0,835       |
| Routes  | 0,834 / 0,823         | 0,797 / 0,819           | 0,813 / 0,854       |
| Murs    | 0,822 / 0,823         | 0,852 / 0,817           | 0,839 / 0,851       |

Lecture prudente : les deux architectures récentes améliorent la détection sur les ponts,
la surface la plus difficile ; chaque modèle n'a été entraîné qu'une fois et la validation
contient peu de photos de ponts, donc l'écart reste à confirmer (plusieurs graines).
Résultats détaillés dans `reports/metrics/` (CSV et JSON).

### Modèle final

Modèle retenu : **ConvNeXt-Tiny**, seuil de décision **0,05** (classe « fissure » si la
probabilité est supérieure ou égale à 0,05 ; les probabilités de ConvNeXt sont basses,
d'où ce seuil). La règle de choix et ses limites sont dans `reports/decision_modele_final.md`.

Mesure unique sur le jeu de test (jamais utilisé pour choisir) : recall 0,832,
précision 0,642, PR-AUC 0,857. Limites : un seul entraînement, seuil sensible, ponts
(surface la plus difficile).

## Lancer l'application avec Docker

Prérequis : Docker Desktop ouvert, et les poids du modèle récupérés avec DVC.

```
dvc pull models/convnext-tiny.onnx.dvc models/convnext-tiny.pt.dvc
docker compose up --build -d
```

| Service  | Adresse               | Rôle                                      |
| -------- | --------------------- | ------------------------------------------ |
| frontend | http://localhost      | interface React (nginx, proxy /api)        |
| backend  | http://127.0.0.1:8000 | API FastAPI : ONNX Runtime CPU et Grad-CAM |

Vérifier et arrêter :

```
docker compose ps
curl.exe http://127.0.0.1:8000/health
docker compose down
```

Les modèles ne sont pas dans les images : ils sont montés en volume depuis `models/`.

Mesure faite dans Docker sur une image de fissure de validation (7001-115.jpg) :
probabilité 0,153 pour un seuil de 0,05, décision « fissure », environ 34 ms par
prédiction sur CPU. Le résultat est identique à celui du backend lancé hors Docker.

## Avancement

| Phase | Contenu                                                      | État                                                                            |
| ----- | ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 1     | Données, index, exploration, DVC, CI minimal                | Fait                                                                             |
| 2     | Chargement des données, métriques, baseline ResNet, MLflow | Fait (baseline PR-AUC val 0,882)                                                 |
| 3     | Modèles avancés, augmentation, Grad-CAM                    | Fait (3.1, Grad-CAM, modèle final ConvNeXt-Tiny) ; augmentation 3.2 optionnelle |
| 4     | Export ONNX, API FastAPI, interface React                    | Fait                                                                             |
| 5     | Docker, docker-compose, CI/CD complet                        | En cours (Docker, compose et CI de tests faits ; DVC partagé à venir)          |
| 6     | Rapport et soutenance                                        | À venir                                                                         |

## Référence

Maguire, M., Dorafshan, S., Thomas, R. J. (2018). *SDNET2018: A concrete crack image
dataset for machine learning applications*. Utah State University.
