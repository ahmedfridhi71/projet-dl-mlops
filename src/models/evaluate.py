import numpy as np
import torch


@torch.no_grad()
def predict(model, loader, device):
    """Retourne (vraies etiquettes, probabilite de fissure) pour tout le loader."""
    model.eval()
    y_true, y_score = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1]
        y_true.extend(y.tolist())
        y_score.extend(probs.cpu().tolist())
    return np.array(y_true), np.array(y_score)