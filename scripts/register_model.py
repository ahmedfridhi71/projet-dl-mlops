"""Model Registry MLflow du modèle "detection-fissures".

On utilise les alias (staging, production), pas les anciens "stages" dépréciés.

Exemples (depuis la racine du dépôt) :
  python -m scripts.register_model --list
  python -m scripts.register_model --register --pt models/convnext-tiny.pt \
      --onnx models/convnext-tiny.onnx --metrics-json reports/metrics/convnext-tiny-r85_val.json
  python -m scripts.register_model --register --run-id <id_du_run>
  python -m scripts.register_model --promote staging              # dernière version
  python -m scripts.register_model --promote production           # la version en staging
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # moins de messages inutiles

from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "detection-fissures"
EXPERIMENT = "fissures-beton-sdnet2018"
ARTIFACT_DIR = "model"  # dossier des artefacts (.pt, .onnx) dans le run
STAGING, PRODUCTION = "staging", "production"


class RegistryError(Exception):
    """Erreur expliquée à l'utilisateur (règle non respectée, fichier manquant...)."""


def get_client(tracking_uri=None):
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _ensure_registered_model(client, name):
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(
            name, description="Classification fissure / pas fissure (SDNET2018).")


def _create_version(client, name, run, pr_auc, threshold, description):
    _ensure_registered_model(client, name)
    tags = {"source_run_name": run.info.run_name}
    if pr_auc is not None:
        tags["val_pr_auc"] = f"{pr_auc:.4f}"
    if threshold is not None:
        tags["threshold"] = str(threshold)
    return client.create_model_version(
        name=name,
        source=f"{run.info.artifact_uri}/{ARTIFACT_DIR}",
        run_id=run.info.run_id,
        tags=tags,
        description=description,
    )


def register_from_files(client, pt, onnx=None, metrics_json=None, run_name=None,
                        name=MODEL_NAME, experiment=EXPERIMENT):
    """Crée un NOUVEAU run (poids + mesures val) puis une nouvelle version du modèle.

    Aucun ancien run n'est modifié.
    """
    pt = Path(pt)
    fichiers = [pt] + ([Path(onnx)] if onnx else [])
    for f in fichiers:
        if not f.exists():
            raise RegistryError(f"Fichier introuvable : {f}")

    mesures, threshold = {}, None
    if metrics_json:
        data = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
        if data.get("split") != "val":
            raise RegistryError(
                f"{metrics_json} n'est pas une mesure sur val (split = {data.get('split')!r}). "
                "Le split test ne sert jamais pour choisir un modèle.")
        threshold = data.get("threshold")
        mesures = {f"val_{k}": float(v) for k, v in data["global"].items()}

    exp = client.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
    run = client.create_run(exp_id, run_name=run_name or f"registry-{pt.stem}",
                            tags={"role": "model-registry"})
    run_id = run.info.run_id
    try:
        client.log_param(run_id, "weights", pt.as_posix())
        if onnx:
            client.log_param(run_id, "onnx", Path(onnx).as_posix())
        if metrics_json:
            client.log_param(run_id, "metrics_json", Path(metrics_json).as_posix())
        if threshold is not None:
            client.log_param(run_id, "threshold", threshold)
        for k, v in mesures.items():
            client.log_metric(run_id, k, v)
        for f in fichiers:
            client.log_artifact(run_id, str(f), ARTIFACT_DIR)
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise

    run = client.get_run(run_id)
    return _create_version(client, name, run, mesures.get("val_pr_auc"), threshold,
                           description=f"Poids : {pt.as_posix()}")


def register_from_run(client, run_id, name=MODEL_NAME):
    """Nouvelle version à partir d'un run existant (le run n'est pas modifié)."""
    try:
        run = client.get_run(run_id)
    except MlflowException:
        raise RegistryError(f"Run introuvable : {run_id}") from None
    noms = [a.path for a in client.list_artifacts(run_id, ARTIFACT_DIR)]
    if not any(n.endswith((".pt", ".onnx")) for n in noms):
        raise RegistryError(
            f"Le run {run_id} n'a pas de .pt ni de .onnx dans le dossier '{ARTIFACT_DIR}'. "
            "Utilisez plutôt --pt (et --onnx, --metrics-json).")
    pr_auc = run_pr_auc(client, run)
    threshold = run.data.params.get("threshold")
    return _create_version(client, name, run, pr_auc, threshold,
                           description=f"Run source : {run.info.run_name}")


def run_pr_auc(client, run):
    """PR-AUC val du MEILLEUR modèle du run (celui dont les poids sont sauvegardés).

    1. best_val_pr_auc (meilleure époque) ;
    2. sinon le maximum de l'historique de val_pr_auc (jamais la dernière époque) ;
    3. sinon global.pr_auc (une seule mesure, faite par export_metrics).
    """
    metrics = run.data.metrics
    if "best_val_pr_auc" in metrics:
        return float(metrics["best_val_pr_auc"])
    if "val_pr_auc" in metrics:
        historique = client.get_metric_history(run.info.run_id, "val_pr_auc")
        return max(float(m.value) for m in historique)
    if "global.pr_auc" in metrics:
        return float(metrics["global.pr_auc"])
    return None


def version_pr_auc(client, mv):
    """PR-AUC val d'une version : lue dans son run, sinon dans son tag val_pr_auc."""
    try:
        pr_auc = run_pr_auc(client, client.get_run(mv.run_id))
    except MlflowException:
        pr_auc = None
    if pr_auc is None and "val_pr_auc" in mv.tags:
        pr_auc = float(mv.tags["val_pr_auc"])
    return pr_auc


def _versions(client, name):
    vs = client.search_model_versions(f"name='{name}'")
    # search_model_versions ne renvoie pas les alias : on relit chaque version
    return sorted((client.get_model_version(name, v.version) for v in vs),
                  key=lambda v: int(v.version))


def get_alias_version(client, alias, name=MODEL_NAME):
    """Version qui porte cet alias, ou None."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def promote(client, alias, version=None, name=MODEL_NAME):
    """Pose l'alias staging ou production. Renvoie (version promue, ancienne version ou None)."""
    versions = _versions(client, name)
    if not versions:
        raise RegistryError(f"Aucune version du modèle {name}. Lancez d'abord --register.")

    if alias == STAGING:
        mv = versions[-1] if version is None else _find(versions, version)
    elif alias == PRODUCTION:
        if version is None:
            mv = get_alias_version(client, STAGING, name)
            if mv is None:
                raise RegistryError("Aucune version n'a l'alias staging : rien à mettre en production.")
        else:
            mv = _find(versions, version)
            if STAGING not in mv.aliases:
                raise RegistryError(
                    f"La version {mv.version} n'a pas l'alias staging. "
                    "Passez-la d'abord en staging (--promote staging --version "
                    f"{mv.version}).")
    else:
        raise RegistryError(f"Alias inconnu : {alias}")

    ancienne = get_alias_version(client, alias, name)
    # Poser l'alias le retire automatiquement de l'ancienne version
    client.set_registered_model_alias(name, alias, mv.version)
    if alias == PRODUCTION:
        # La version quitte staging : elle est maintenant en production
        client.delete_registered_model_alias(name, STAGING)
    if ancienne is not None and int(ancienne.version) == int(mv.version):
        ancienne = None
    return client.get_model_version(name, mv.version), ancienne


def _find(versions, version):
    for v in versions:
        if int(v.version) == int(version):
            return v
    raise RegistryError(f"Version {version} introuvable.")


def list_versions(client, name=MODEL_NAME):
    """Lignes (version, alias, pr_auc, run_id, nom du run) pour --list."""
    lignes = []
    for mv in _versions(client, name):
        try:
            run_name = client.get_run(mv.run_id).info.run_name
        except MlflowException:
            run_name = "?"
        lignes.append({
            "version": int(mv.version),
            "aliases": sorted(mv.aliases),
            "val_pr_auc": version_pr_auc(client, mv),
            "run_id": mv.run_id,
            "run_name": run_name,
        })
    return lignes


def print_list(client, name=MODEL_NAME):
    lignes = list_versions(client, name)
    if not lignes:
        print(f"Aucune version pour le modèle {name}.")
        return
    print(f"Modèle : {name}")
    print(f"{'version':>7}  {'alias':<20}  {'PR-AUC val':>10}  run source")
    for li in lignes:
        alias = ", ".join(li["aliases"]) or "-"
        pr = f"{li['val_pr_auc']:.4f}" if li["val_pr_auc"] is not None else "?"
        print(f"{li['version']:>7}  {alias:<20}  {pr:>10}  {li['run_name']} ({li['run_id']})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Model Registry MLflow (alias staging / production).")
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true", help="créer une nouvelle version")
    action.add_argument("--promote", choices=[STAGING, PRODUCTION], help="poser un alias")
    action.add_argument("--list", action="store_true", help="afficher les versions")
    p.add_argument("--version", type=int, help="version à promouvoir (par défaut : voir l'aide)")
    p.add_argument("--run-id", help="--register à partir d'un run existant")
    p.add_argument("--pt", help="--register : poids PyTorch (crée un nouveau run)")
    p.add_argument("--onnx", help="--register : modèle ONNX (optionnel)")
    p.add_argument("--metrics-json", help="--register : mesures val (export_metrics)")
    p.add_argument("--run-name", help="--register : nom du nouveau run")
    p.add_argument("--name", default=MODEL_NAME)
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--tracking-uri", default=None, help=f"par défaut : {TRACKING_URI}")
    args = p.parse_args(argv)

    client = get_client(args.tracking_uri)
    try:
        if args.register:
            if args.run_id:
                mv = register_from_run(client, args.run_id, args.name)
            elif args.pt:
                mv = register_from_files(client, args.pt, args.onnx, args.metrics_json,
                                         args.run_name, args.name, args.experiment)
            else:
                raise RegistryError("--register demande --run-id ou --pt.")
            print(f"Nouvelle version : {args.name} v{mv.version} (run {mv.run_id})")
        elif args.promote:
            mv, ancienne = promote(client, args.promote, args.version, args.name)
            print(f"{args.name} v{mv.version} a maintenant l'alias {args.promote}.")
            if ancienne is not None:
                print(f"L'ancienne version v{ancienne.version} perd l'alias {args.promote}.")
        else:
            print_list(client, args.name)
    except RegistryError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
