import argparse

import torch

from src.data.loaders import get_loaders
from src.models.baseline import build_resnet18
from src.models.evaluate import predict
from src.utils.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = get_loaders(batch_size=64)
    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    y_true, y_score = predict(model, val_loader, device)

    print(f"Modele : {args.weights}")
    print(f"{'seuil':>6} {'recall':>8} {'precision':>10} {'f1':>7}")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        m = compute_metrics(y_true, y_score, threshold=t)
        print(f"{t:>6.1f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>7.3f}")


if __name__ == "__main__":
    main()