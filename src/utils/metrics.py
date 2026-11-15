import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def compute_metrics(y_true, y_score, threshold=0.5):
    """Métriques de la classe fissure.

    y_true  : étiquettes réelles (0 = sain, 1 = fissure)
    y_score : probabilité de fissure prédite par le modèle (entre 0 et 1)
    threshold : au-dessus de ce seuil, le modèle annonce « fissure »
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pr_auc = average_precision_score(y_true, y_score) if y_true.sum() > 0 else 0.0

    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_by_surface(y_true, y_score, surfaces, threshold=0.5):
    """Mêmes métriques, séparément pour chaque surface (D, P, W)."""
    y_true, y_score, surfaces = map(np.asarray, (y_true, y_score, surfaces))
    return {s: compute_metrics(y_true[surfaces == s], y_score[surfaces == s], threshold)
            for s in np.unique(surfaces)}