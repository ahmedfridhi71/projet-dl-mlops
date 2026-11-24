import argparse
import json
from pathlib import Path

import torch

from src.data.loaders import get_loaders
from src.models.architecture import build_model
from src.models.evaluate import predict
from src.utils.metrics import compute_metrics, metrics_by_surface


def to_python(o):
    """Convertit les nombres numpy en nombres Python pour l'ecriture JSON."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Type non supporte : {type(o)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "efficientnet_v2_s", "convnext_tiny"])
    p.add_argument("--name", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = get_loaders(batch_size=64)
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    y_true, y_score = predict(model, loader, device)
    surfaces = loader.dataset.df["surface"].values

    result = {
        "name": args.name,
        "model": args.model,
        "weights": args.weights,
        "split": args.split,
        "threshold": args.threshold,
        "n_images": int(len(y_true)),
        "global": compute_metrics(y_true, y_score, args.threshold),
        "by_surface": metrics_by_surface(y_true, y_score, surfaces, args.threshold),
    }

    out = Path("reports/metrics") / f"{args.name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=to_python), encoding="utf-8")
    print("Resultats ecrits dans", out)


if __name__ == "__main__":
    main()