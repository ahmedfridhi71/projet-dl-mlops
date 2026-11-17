import argparse
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from src.data.loaders import get_loaders
from src.models.baseline import build_resnet18
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pos-weight", type=float, default=1.0)  # NOUVEAU (changement 1)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--run-name", default="resnet18-baseline")
    p.add_argument("--out", default="models/baseline.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil :", device)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baseline-sdnet")

    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)
    model = build_resnet18().to(device)
    class_weights = torch.tensor([1.0, args.pos_weight], device=device)  # NOUVEAU (changement 2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)                # NOUVEAU (changement 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": "resnet18",
            "pretrained": True,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "pos_weight": args.pos_weight,  # NOUVEAU (changement 3)
            "fp16": device.type == "cuda",
            "max_batches": args.max_batches,
        })

        best = -1.0
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
            if m["pr_auc"] > best:
                best = m["pr_auc"]
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.out)
                print("  -> meilleur modele sauvegarde")

        mlflow.log_metric("best_val_pr_auc", best)
        mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()