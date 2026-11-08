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


def test_processed_suivi_par_dvc():
    out = _read_dvc("data/processed.dvc")
    assert out["path"] == "processed"
    assert out["nfiles"] >= 1