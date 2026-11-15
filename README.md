
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
.github/workflows/   CI (tests) et build Docker (à compléter)
data/                données, versionnées par DVC (non présentes sur GitHub)
notebooks/           exploration du dataset
reports/             figures et rapport de qualité des données
src/data/            index et découpage, contrôle qualité
tests/               tests automatiques (pytest)
backend/             API FastAPI (à venir)
frontend/            interface React (à venir)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install dvc pandas numpy pillow matplotlib pytest pyyaml
```

Un fichier `requirements.txt` figé sera ajouté avec le pipeline d'entraînement.

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

# Contrôle qualité (images corrompues, doublons exacts, planche d'exemples)
.\.venv\Scripts\python.exe src\data\check_quality.py

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Qualité des données (résultats du contrôle)

- 0 image corrompue ; toutes les images font 256x256 en RGB.
- 13 groupes de doublons exacts (0,02 %), dont 2 avec des étiquettes contradictoires ;
  ils seront exclus au chargement des données.
- Aucun doublon réparti sur plusieurs splits (aucune fuite).

## Avancement

| Phase | Contenu                                                      | État    |
| ----- | ------------------------------------------------------------ | -------- |
| 1     | Données, index, exploration, DVC, CI minimal                | Fait     |
| 2     | Chargement des données, métriques, baseline ResNet, MLflow | À venir |
| 3     | Modèles avancés, augmentation, Grad-CAM                    | À venir |
| 4     | Export ONNX, API FastAPI, interface React                    | À venir |
| 5     | Docker, docker-compose, CI/CD complet                        | À venir |
| 6     | Rapport et soutenance                                        | À venir |

## Référence

Maguire, M., Dorafshan, S., Thomas, R. J. (2018). *SDNET2018: A concrete crack image
dataset for machine learning applications*. Utah State University.
