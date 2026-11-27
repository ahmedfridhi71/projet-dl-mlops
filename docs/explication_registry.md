# Le Model Registry MLflow, expliqué simplement

## 1. C'est quoi ?

MLflow garde déjà la trace de chaque **run** (un entraînement ou une mesure).
Le **Model Registry** ajoute une liste officielle des modèles qu'on peut utiliser.

Notre modèle s'appelle **`detection-fissures`**.

## 2. Les versions

Chaque fois qu'on enregistre un modèle, MLflow crée une **nouvelle version** : v1, v2, v3...

- Une version ne change jamais : elle pointe vers un run et ses fichiers (`.pt`, `.onnx`).
- On garde l'historique : on sait d'où vient chaque version (le **run source**).
- On peut revenir à une ancienne version en cas de problème.

Pour chaque version, le script garde aussi la **PR-AUC val** (tag `val_pr_auc`) et le seuil.

## 3. Les alias

Un **alias**, c'est une étiquette posée sur **une** version. Nous en utilisons deux :

| Alias | Sens |
|---|---|
| `staging` | Version candidate : on la vérifie avant de la mettre en service. |
| `production` | Version en service : c'est celle qu'il faut déployer. |

Un alias est sur une seule version à la fois.
Quand on pose `production` sur la v2, la v1 **perd** automatiquement l'alias.

On peut charger une version par son alias, sans connaître son numéro :
`models:/detection-fissures@production`.

Note : MLflow avait avant des « stages » (Staging, Production). Ils sont **dépréciés** depuis MLflow 2.9.
Nous utilisons les alias, qui sont la méthode recommandée aujourd'hui.

## 4. Pourquoi staging puis production ?

On ne met pas un nouveau modèle en service directement. On passe par deux marches :

1. **staging** : le modèle est candidat. On regarde ses mesures sur val, on le teste (API, Grad-CAM...).
2. **production** : seulement si tout va bien.

Le script **refuse** de mettre en production une version qui n'est pas en staging.
C'est un garde-fou : aucun modèle n'arrive en service sans avoir été candidat.

Quand une version passe en production, elle quitte staging (l'alias `staging` est retiré).

## 5. Les commandes

Toutes les commandes se lancent depuis la racine du dépôt.

Voir les versions, leurs alias, leur PR-AUC val et leur run source :

```powershell
.\.venv\Scripts\python.exe -m scripts.register_model --list
```

Créer une nouvelle version à partir de fichiers.
Le script crée un **nouveau run** dans l'expérience `fissures-beton-sdnet2018`. Les anciens runs ne sont pas modifiés :

```powershell
.\.venv\Scripts\python.exe -m scripts.register_model --register `
  --pt models/convnext-tiny.pt --onnx models/convnext-tiny.onnx `
  --metrics-json reports/metrics/convnext-tiny-r85_val.json --run-name convnext-tiny-registry
```

Le fichier `--metrics-json` doit être une mesure sur **val**. Une mesure sur test est refusée.

Créer une nouvelle version à partir d'un run existant (il doit avoir ses fichiers dans le dossier `model/`) :

```powershell
.\.venv\Scripts\python.exe -m scripts.register_model --register --run-id <id_du_run>
```

Passer la dernière version en staging, puis en production :

```powershell
.\.venv\Scripts\python.exe -m scripts.register_model --promote staging
.\.venv\Scripts\python.exe -m scripts.register_model --promote production
```

Choisir une version précise : ajouter `--version 2`.

Voir le Registry dans l'interface MLflow :

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir http://127.0.0.1:5000, onglet **Models**.

## 6. Les tests

`tests/test_registry.py` vérifie les règles avec une base MLflow **temporaire** et de faux petits fichiers.
Il n'y a pas besoin de GPU ni des vrais poids. Ces tests sont sautés si MLflow n'est pas installé (c'est le cas dans la CI).
