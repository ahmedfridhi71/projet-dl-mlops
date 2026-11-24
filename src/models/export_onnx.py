import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.data.loaders import get_loaders
from src.models.architecture import build_model


def softmax_1(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/convnext-tiny.pt")
    p.add_argument("--model", default="convnext_tiny",
                   choices=["resnet18", "efficientnet_v2_s", "convnext_tiny"])
    p.add_argument("--out", default="models/convnext-tiny.onnx")
    p.add_argument("--threshold", type=float, default=0.05)
    args = p.parse_args()

    # 1. Export
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    taille = Path(args.out).stat().st_size / 1e6
    print(f"ONNX ecrit : {args.out} ({taille:.1f} Mo)")

    # 2. Verification : memes probabilites que PyTorch ?
    _, val_loader, _ = get_loaders(batch_size=32)
    ds = val_loader.dataset
    labels = np.array(ds.labels)
    idx = list(np.where(labels == 1)[0][:16]) + list(np.where(labels == 0)[0][:16])
    x = torch.stack([ds[int(i)][0] for i in idx])

    with torch.no_grad():
        p_torch = torch.softmax(model(x), dim=1)[:, 1].numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    p_onnx = softmax_1(sess.run(["logits"], {"image": x.numpy()})[0])

    ecart = float(np.abs(p_torch - p_onnx).max())
    memes = int(((p_torch >= args.threshold) == (p_onnx >= args.threshold)).sum())
    print(f"Ecart maximal des probabilites (PyTorch contre ONNX) : {ecart:.2e}")
    print(f"Decisions identiques au seuil {args.threshold} : {memes} / {len(idx)}")

    # 3. Vitesse sur une image (CPU)
    x1 = x[:1]
    xn = x1.numpy()
    with torch.no_grad():
        for _ in range(3):
            model(x1)
        t = time.perf_counter()
        for _ in range(20):
            model(x1)
        ms_torch = (time.perf_counter() - t) / 20 * 1000
    for _ in range(3):
        sess.run(None, {"image": xn})
    t = time.perf_counter()
    for _ in range(20):
        sess.run(None, {"image": xn})
    ms_onnx = (time.perf_counter() - t) / 20 * 1000
    print(f"Vitesse par image (CPU) : PyTorch {ms_torch:.1f} ms, ONNX Runtime {ms_onnx:.1f} ms")


if __name__ == "__main__":
    main()