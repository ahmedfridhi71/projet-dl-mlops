from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_dvc(relative_path):
    """Lit une fiche .dvc et renvoie sa première sortie suivie."""
    fiche = ROOT / relative_path
    assert fiche.exists(), f"Fiche DVC introuvable : {relative_path}"
    return yaml.safe_load(fiche.read_text(encoding="utf-8"))["outs"][0]


def test_images_suivies_par_dvc():
    out = _read_dvc("data/raw/SDNET2018_images.dvc")
    assert out["path"] == "SDNET2018_images"
    assert out["nfiles"] == 56092, "Le dataset doit contenir 56 092 images"


def _read_yaml(relative_path):
    """Lit un fichier YAML du dépôt (dvc.yaml, dvc.lock)."""
    fichier = ROOT / relative_path
    assert fichier.exists(), f"Fichier introuvable : {relative_path}"
    return yaml.safe_load(fichier.read_text(encoding="utf-8"))


def test_processed_suivi_par_dvc():
    """data/processed est produit par le stage "prepare" du pipeline DVC."""
    attendues = ["data/processed/index.csv", "data/processed/split_report.txt"]

    stage = _read_yaml("dvc.yaml")["stages"].get("prepare")
    assert stage is not None, "Le stage prepare manque dans dvc.yaml"
    assert stage["outs"] == attendues

    verrou = _read_yaml("dvc.lock")["stages"].get("prepare")
    assert verrou is not None, "Le stage prepare manque dans dvc.lock"
    empreintes = {out["path"]: out.get("md5") for out in verrou["outs"]}
    for chemin in attendues:
        assert empreintes.get(chemin), f"Pas d'empreinte md5 pour {chemin} dans dvc.lock"