import argparse
import csv
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from src.data.loaders import get_loaders
from src.models.architecture import build_model  # NOUVEAU (changement 1)
from src.models.evaluate import predict
from src.utils.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=None):
    model.train()
    loss_sum, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / n


def save_history(rows, path):
    """Ecrit une ligne par epoque dans un fichier CSV (reecrit a chaque epoque)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "efficientnet_v2_s", "convnext_tiny"])  # NOUVEAU (changement 2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = Path(args.out) if args.out else Path("models") / f"{args.run_name}.pt"
    history_path = Path("reports/metrics") / f"{args.run_name}_epochs.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil :", device)
    print("Modele :", args.model)
    print("Echantillonnage equilibre :", args.balanced)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baseline-sdnet")

    train_loader, val_loader, _ = get_loaders(
        batch_size=args.batch_size, balanced=args.balanced
    )
    model = build_model(args.model).to(device)  # NOUVEAU (changement 3)
    class_weights = torch.tensor([1.0, args.pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": args.model,  # NOUVEAU : le vrai nom du modele
            "pretrained": True,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "pos_weight": args.pos_weight,
            "balanced": args.balanced,
            "fp16": device.type == "cuda",
            "max_batches": args.max_batches,
        })

        rows = []
        best, best_epoch, best_m = -1.0, 0, None
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                   scaler, device, args.max_batches)
            y_true, y_score = predict(model, val_loader, device)
            m = compute_metrics(y_true, y_score)
            duree = time.time() - t0

            print(f"Epoque {epoch} | loss {loss:.4f} | val recall {m['recall']:.3f} "
                  f"| precision {m['precision']:.3f} | f1 {m['f1']:.3f} "
                  f"| PR-AUC {m['pr_auc']:.3f} | {duree:.0f}s")

            mlflow.log_metrics({
                "train_loss": loss,
                "val_recall": m["recall"],
                "val_precision": m["precision"],
                "val_f1": m["f1"],
                "val_pr_auc": m["pr_auc"],
                "epoch_time_s": duree,
            }, step=epoch)

            rows.append({
                "epoch": epoch,
                "train_loss": round(loss, 4),
                "val_recall": round(float(m["recall"]), 4),
                "val_precision": round(float(m["precision"]), 4),
                "val_f1": round(float(m["f1"]), 4),
                "val_pr_auc": round(float(m["pr_auc"]), 4),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
                "tn": int(m["tn"]),
                "epoch_time_s": round(duree),
            })
            save_history(rows, history_path)

            if m["pr_auc"] > best:
                best, best_epoch, best_m = m["pr_auc"], epoch, m
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out)
                print("  -> meilleur modele sauvegarde")

        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_val_recall": float(best_m["recall"]),
            "best_val_precision": float(best_m["precision"]),
            "best_val_f1": float(best_m["f1"]),
            "best_val_pr_auc": float(best),
        })
        mlflow.log_artifact(str(out))
        mlflow.log_artifact(str(history_path))

        print(f"Meilleure epoque : {best_epoch} (PR-AUC {best:.3f})")
        print("Historique :", history_path)
        print("Modele     :", out)


if __name__ == "__main__":
    main()