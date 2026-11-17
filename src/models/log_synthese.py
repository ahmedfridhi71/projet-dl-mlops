import json
from pathlib import Path

import mlflow

DIR = Path("reports/metrics")
MODELES = ["resnet18-baseline", "resnet18-poids3", "resnet18-balanced"]


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baseline-sdnet")

    with mlflow.start_run(run_name="synthese-phase2"):
        # Chiffres principaux de chaque modele (seuil 0.5, validation)
        for nom in MODELES:
            data = json.loads((DIR / f"{nom}_val.json").read_text(encoding="utf-8"))
            g = data["global"]
            mlflow.log_metrics({
                f"{nom}_pr_auc": g["pr_auc"],
                f"{nom}_recall": g["recall"],
                f"{nom}_precision": g["precision"],
                f"{nom}_f1": g["f1"],
            })

        # Fichiers de resultats joints a l'essai
        mlflow.log_artifact(str(DIR / "synthese.csv"), artifact_path="resultats")
        for nom in MODELES:
            mlflow.log_artifact(str(DIR / f"{nom}_seuils.csv"), artifact_path="resultats")
            mlflow.log_artifact(str(DIR / f"{nom}_val.json"), artifact_path="resultats")

    print("Essai 'synthese-phase2' ajoute dans MLflow")


if __name__ == "__main__":
    main()