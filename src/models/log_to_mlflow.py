"""Enregistre dans MLflow les mesures déjà calculées (reports/metrics/*_val.json et *_test.json).

Un run par fichier JSON. Rien n'est réentraîné et rien n'est recalculé.
Relancer le script ne crée pas de doublon : un run déjà présent est sauté.
"""
import json
import math
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIENCE = "fissures-beton-sdnet2018"
DOSSIER = Path("reports/metrics")


def aplatir(d, prefixe=""):
    """{'global': {'recall': 0.8}} -> {'global.recall': 0.8} (nombres seulement)."""
    sortie = {}
    for cle, val in d.items():
        nom = f"{prefixe}{cle}"
        if isinstance(val, dict):
            sortie.update(aplatir(val, nom + "."))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            if math.isfinite(val):
                sortie[nom] = float(val)
    return sortie


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIENCE)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIENCE)

    # Supprime le run d'exemple écrit à la main (pas un vrai résultat).
    runs = client.search_runs([exp.experiment_id])
    for r in runs:
        if r.info.run_name == "test-experience":
            client.delete_run(r.info.run_id)
            print("supprimé : test-experience")

    deja = {r.info.run_name for r in client.search_runs([exp.experiment_id])}

    fichiers = sorted(DOSSIER.glob("*_val.json")) + sorted(DOSSIER.glob("*_test.json"))
    for f in fichiers:
        nom_run = f.stem
        if nom_run in deja:
            print("déjà présent :", nom_run)
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        with mlflow.start_run(run_name=nom_run):
            for cle in ("model", "split", "threshold", "weights", "n_images"):
                if cle in data:
                    mlflow.log_param(cle, data[cle])
            metriques = aplatir({k: data[k] for k in ("global", "by_surface") if k in data})
            mlflow.log_metrics(metriques)
            mlflow.set_tag("source", str(f).replace("\\", "/"))
            mlflow.log_artifact(str(f))
        print("enregistré :", nom_run, f"({len(metriques)} métriques)")


if __name__ == "__main__":
    main()