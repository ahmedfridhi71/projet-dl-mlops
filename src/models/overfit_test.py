import torch
import torch.nn as nn

from src.data.loaders import get_loaders
from src.models.baseline import build_resnet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil :", device)

    train_loader, _, _ = get_loaders(batch_size=32)
    # On garde seulement 2 lots de 32 images = 64 images fixes
    it = iter(train_loader)
    lots = [next(it), next(it)]

    model = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 16):
        total, correct, loss_sum = 0, 0, 0.0
        model.train()
        for x, y in lots:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"Epoque {epoch:2d} | loss {loss_sum/total:.4f} | acc {correct/total:.2%}")


if __name__ == "__main__":
    main()