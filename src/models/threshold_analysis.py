import argparse
import csv
from pathlib import Path

import torch

from src.data.loaders import get_loaders
from src.models.architecture import build_model
from src.models.evaluate import predict
from src.utils.metrics import compute_metrics

SEUILS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEUILS_FINS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "efficientnet_v2_s", "convnext_tiny"])
    p.add_argument("--name", default=None)
    p.add_argument("--fine", action="store_true",
                   help="seuils plus bas (0.005 a 0.5), fichier *_seuils_fins.csv")
    args = p.parse_args()

    name = args.name or Path(args.weights).stem
    suffixe = "_seuils_fins" if args.fine else "_seuils"
    out = Path("reports/metrics") / f"{name}{suffixe}.csv"
    seuils = SEUILS_FINS if args.fine else SEUILS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = get_loaders(batch_size=64)
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    y_true, y_score = predict(model, val_loader, device)

    rows = []
    print(f"Modele : {args.weights} ({args.model})")
    print(f"{'seuil':>6} {'recall':>8} {'precision':>10} {'f1':>7}")
    for t in seuils:
        m = compute_metrics(y_true, y_score, threshold=t)
        print(f"{t:>6.3f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>7.3f}")
        rows.append({
            "seuil": t,
            "recall": round(float(m["recall"]), 4),
            "precision": round(float(m["precision"]), 4),
            "f1": round(float(m["f1"]), 4),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            "tn": int(m["tn"]),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Fichier ecrit :", out)


if __name__ == "__main__":
    main()