import sys
import time

import torch
import torch.nn as nn

from src.models.architecture import build_model

LOTS_PAR_EPOQUE = 1227  # environ 39 260 images / 32


def bench(name, batch_size, steps=10):
    device = torch.device("cuda")
    model = build_model(name, pretrained=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)

    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 2):
        if i == 2:  # on ignore les 2 premiers pas (demarrage)
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad()
        with torch.autocast(device_type="cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    dt = (time.time() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    minutes = dt * LOTS_PAR_EPOQUE * (32 / batch_size) / 60
    print(f"{name:18s} batch {batch_size:2d} | {dt * 1000:4.0f} ms/lot | "
          f"memoire max {mem:.2f} Go | environ {minutes:.1f} min/epoque (GPU seul)")


def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    for name in ["resnet18", "efficientnet_v2_s", "convnext_tiny"]:
        try:
            bench(name, batch_size)
        except torch.cuda.OutOfMemoryError:
            print(f"{name:18s} batch {batch_size:2d} | MEMOIRE INSUFFISANTE")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()