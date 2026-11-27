"""Tests du Model Registry (scripts/register_model.py).

Sans GPU et sans les vrais poids : base sqlite temporaire et faux petits fichiers.
Sautés si mlflow n'est pas installé (cas de la CI).
"""
import json

import pytest

pytest.importorskip("mlflow")

from scripts import register_model as rm  # noqa: E402


@pytest.fixture
def client(tmp_path):
    """Base MLflow vide, rangée dans un dossier temporaire."""
    c = rm.get_client(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    c.create_experiment(rm.EXPERIMENT, artifact_location=(tmp_path / "mlruns").as_uri())
    return c


@pytest.fixture
def fichiers(tmp_path):
    """Faux poids, faux ONNX et une fausse mesure sur val."""
    pt = tmp_path / "faux.pt"
    pt.write_bytes(b"pas de vrais poids")
    onnx = tmp_path / "faux.onnx"
    onnx.write_bytes(b"pas de vrai onnx")
    mesure = tmp_path / "faux_val.json"
    mesure.write_text(json.dumps({
        "split": "val", "threshold": 0.05,
        "global": {"pr_auc": 0.9, "recall": 0.87, "precision": 0.7},
    }), encoding="utf-8")
    return pt, onnx, mesure


def _register(client, fichiers):
    pt, onnx, mesure = fichiers
    return rm.register_from_files(client, pt, onnx, mesure)


def test_register_cree_un_run_et_une_version(client, fichiers):
    mv = _register(client, fichiers)
    assert int(mv.version) == 1
    assert mv.tags["val_pr_auc"] == "0.9000"
    run = client.get_run(mv.run_id)
    assert run.data.metrics["val_pr_auc"] == 0.9
    noms = {a.path for a in client.list_artifacts(mv.run_id, rm.ARTIFACT_DIR)}
    assert noms == {"model/faux.pt", "model/faux.onnx"}


def _faux_run_entrainement(client, fichiers, historique, best=None):
    """Faux run comme train.py : val_pr_auc par époque, poids dans model/."""
    pt, _, _ = fichiers
    exp_id = client.get_experiment_by_name(rm.EXPERIMENT).experiment_id
    run_id = client.create_run(exp_id, run_name="faux-train").info.run_id
    for epoque, valeur in enumerate(historique, 1):
        client.log_metric(run_id, "val_pr_auc", valeur, step=epoque)
    if best is not None:
        client.log_metric(run_id, "best_val_pr_auc", best)
    client.log_artifact(run_id, str(pt), rm.ARTIFACT_DIR)
    client.set_terminated(run_id, "FINISHED")
    return run_id


def test_tag_meilleure_epoque_avec_best(client, fichiers):
    # Meilleure époque : 0.92 (époque 2). Dernière époque : 0.85.
    run_id = _faux_run_entrainement(client, fichiers, [0.80, 0.92, 0.85], best=0.92)
    mv = rm.register_from_run(client, run_id)
    assert mv.tags["val_pr_auc"] == "0.9200"
    assert rm.list_versions(client)[0]["val_pr_auc"] == 0.92


def test_tag_meilleure_epoque_sans_best(client, fichiers):
    # Pas de best_val_pr_auc : on prend le maximum de l'historique, pas la dernière valeur.
    run_id = _faux_run_entrainement(client, fichiers, [0.80, 0.92, 0.85])
    mv = rm.register_from_run(client, run_id)
    assert mv.tags["val_pr_auc"] == "0.9200"
    assert rm.list_versions(client)[0]["val_pr_auc"] == 0.92


def test_register_refuse_le_split_test(client, fichiers, tmp_path):
    pt, onnx, _ = fichiers
    mesure_test = tmp_path / "faux_test.json"
    mesure_test.write_text(json.dumps({"split": "test", "global": {"pr_auc": 0.8}}),
                           encoding="utf-8")
    with pytest.raises(rm.RegistryError, match="test"):
        rm.register_from_files(client, pt, onnx, mesure_test)


def test_production_exige_staging(client, fichiers):
    _register(client, fichiers)
    with pytest.raises(rm.RegistryError, match="staging"):
        rm.promote(client, rm.PRODUCTION)               # aucune version en staging
    with pytest.raises(rm.RegistryError, match="staging"):
        rm.promote(client, rm.PRODUCTION, version=1)    # v1 n'est pas en staging


def test_promotion_deplace_les_alias(client, fichiers):
    _register(client, fichiers)
    rm.promote(client, rm.STAGING)
    mv, ancienne = rm.promote(client, rm.PRODUCTION)
    assert int(mv.version) == 1 and ancienne is None
    assert mv.aliases == [rm.PRODUCTION]                # v1 quitte staging

    _register(client, fichiers)                         # v2
    rm.promote(client, rm.STAGING)
    mv, ancienne = rm.promote(client, rm.PRODUCTION)
    assert int(mv.version) == 2 and int(ancienne.version) == 1

    lignes = {li["version"]: li for li in rm.list_versions(client)}
    assert lignes[1]["aliases"] == []                   # l'ancienne perd son alias
    assert lignes[2]["aliases"] == [rm.PRODUCTION]
    assert lignes[2]["val_pr_auc"] == 0.9
    assert rm.get_alias_version(client, rm.STAGING) is None
