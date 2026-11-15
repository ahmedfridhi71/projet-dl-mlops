import numpy as np
import pytest

from src.utils.metrics import compute_metrics, metrics_by_surface


def test_modele_toujours_sain():
    # 85 patchs sains, 15 fissures : un modèle qui répond toujours « sain »
    y_true = np.array([0] * 85 + [1] * 15)
    m = compute_metrics(y_true, np.zeros(100))
    assert m["accuracy"] == pytest.approx(0.85)
    assert m["recall"] == 0.0
    assert m["tp"] == 0 and m["fn"] == 15


def test_modele_parfait():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    m = compute_metrics(y_true, y_true.astype(float))
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0
    assert m["pr_auc"] == pytest.approx(1.0)


def test_cas_calcule_a_la_main():
    # 4 vraies fissures : le modèle en trouve 3, en rate 1, et fait 1 fausse alerte
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.1, 0.1])
    m = compute_metrics(y_true, y_score)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_par_surface():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.1, 0.1])
    surfaces = np.array(["D", "D", "W", "W"])
    res = metrics_by_surface(y_true, y_score, surfaces)
    assert res["D"]["recall"] == 1.0
    assert res["W"]["recall"] == 0.0