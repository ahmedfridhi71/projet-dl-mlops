# Le pipeline DVC, expliqué simplement

## 1. C'est quoi un pipeline DVC ?

Un pipeline, c'est une suite d'étapes. Chaque étape s'appelle un **stage**.

Pour chaque stage, on écrit dans `dvc.yaml` :

- la **commande** à lancer (`cmd`) ;
- ce dont elle a besoin (`deps`, pour « dépendances ») : données, scripts ;
- ce qu'elle produit (`outs`, `metrics`, `plots`) ;
- les **paramètres** qu'elle lit dans `params.yaml`.

DVC garde une empreinte (md5) de tout cela dans `dvc.lock`.
Quand on lance `dvc repro`, DVC compare les empreintes :

- rien n'a changé → le stage est **sauté** (gain de temps) ;
- une dépendance ou un paramètre a changé → le stage est **relancé**, ainsi que tous les stages qui viennent après.

C'est ce qui rend le projet **reproductible** : une seule commande refait tout ce qui doit l'être, et seulement ça.

## 2. Les fichiers

| Fichier | Rôle |
|---|---|
| `dvc.yaml` | La liste des stages (les étapes). |
| `params.yaml` | Les réglages (époques, learning rate, seuil...). |
| `dvc.lock` | Les empreintes du dernier passage. Écrit par DVC, ne pas modifier à la main. |

Ces trois fichiers vont dans Git. Les données et les poids, non : ils restent dans le cache DVC.

## 3. Les 3 stages

```
data/raw/SDNET2018_images.dvc
            │
         prepare  ──►  data/processed/index.csv, split_report.txt
            │
          train   ──►  models/pipeline/pipeline-convnext-tiny.pt
            │          reports/metrics/pipeline-convnext-tiny_epochs.csv
         evaluate ──►  reports/metrics/pipeline-convnext-tiny_val.json
```

### prepare

- Lance `src/data/make_index.py` : il liste les 56 092 images et fait le découpage train / val / test **par photo** (graine 42).
- Puis lance `src/data/flag_exclusions.py` : il repère les doublons et ajoute la colonne `exclure`.
- Sorties : `data/processed/index.csv` et `data/processed/split_report.txt`.

Avant, ce dossier était suivi par `data/processed.dvc`. Ce fichier a été remplacé par ce stage.
Le nouvel `index.csv` a exactement le même md5 que l'ancien (`b393c36f...`) : le résultat est identique.

Remarque : `flag_exclusions.py` réécrit aussi `reports/quality/exclusions.csv`. Ce fichier est suivi par Git, pas par DVC. Son contenu ne change pas.

### train

- Lance `src/models/train.py` avec les valeurs de la section `train` de `params.yaml`.
- Sortie : les poids dans `models/pipeline/pipeline-convnext-tiny.pt`.
  Les poids du modèle en service (`models/convnext-tiny.pt`) ne sont **jamais** écrasés.
- Le script écrit aussi les courbes par époque dans `reports/metrics/pipeline-convnext-tiny_epochs.csv` (déclaré en `plots`), et crée un run dans `mlflow.db` (expérience `baseline-sdnet`).
- Durée : 20 à 30 minutes avec un GPU.

### evaluate

- Lance `src/models/export_metrics.py` sur le split **val** uniquement, au seuil 0.05.
- Sortie : `reports/metrics/pipeline-convnext-tiny_val.json` (déclaré en `metrics`).
- Le split **test** n'est jamais utilisé : il a déjà servi une fois pour la mesure finale.
  Ne mettez jamais `split: test` dans `params.yaml`.

Les noms `pipeline-...` sont uniques : les fichiers déjà présents dans `reports/metrics/` ne sont jamais touchés.

## 4. Les commandes

Voir le graphe des stages :

```powershell
.\.venv\Scripts\dvc.exe dag
```

Voir ce qui a changé depuis le dernier passage :

```powershell
.\.venv\Scripts\dvc.exe status
```

Voir ce que `dvc repro` ferait, **sans rien lancer** :

```powershell
.\.venv\Scripts\dvc.exe repro --dry
```

Tout lancer, avec **une seule commande** (attention : entraînement GPU de 20 à 30 minutes) :

```powershell
.\.venv\Scripts\dvc.exe repro
```

Lancer un seul stage (et ceux dont il dépend) :

```powershell
.\.venv\Scripts\dvc.exe repro prepare
```

Voir les métriques après un passage :

```powershell
.\.venv\Scripts\dvc.exe metrics show
```

Envoyer les nouvelles sorties vers le stockage DVC :

```powershell
.\.venv\Scripts\dvc.exe push
```

Note : avec `repro --dry`, DVC peut afficher une erreur sur `evaluate` quand les poids de `train` n'existent pas encore. C'est normal : en mode à blanc, `train` n'est pas vraiment lancé, donc le fichier attendu par `evaluate` manque.

## 5. Exemple : changer un paramètre

1. Dans `params.yaml`, mettre `epochs: 6`.
2. `dvc status` montre que `train` a changé.
3. `dvc repro` relance `train` puis `evaluate`. `prepare` est sauté, car les données n'ont pas changé.

## 6. Limites de reproductibilité

- **Pas de graine pour l'entraînement.** `src/models/train.py` n'a pas d'option `--seed`.
  L'initialisation de la tête du modèle, l'ordre des images et le GPU ont une part de hasard.
  Deux entraînements avec les mêmes paramètres donnent donc des résultats **proches, mais pas identiques**.
- **La graine du découpage est écrite en dur.** `prepare.seed: 42` dans `params.yaml` est seulement une information :
  la valeur utilisée est celle de `src/data/make_index.py`. Changer `params.yaml` ne change pas le découpage.
- **Le run MLflow n'est pas une sortie DVC.** `train` ajoute un run dans `mlflow.db`, mais DVC ne le suit pas.
- **`python -X utf8`** : cette option force l'UTF-8. Sans elle, sous Windows, le dernier message des scripts (avec ✅) peut faire planter la commande, et DVC croirait que le stage a échoué.
