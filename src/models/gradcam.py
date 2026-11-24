import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.loaders import get_loaders
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.architecture import build_model
from src.models.evaluate import predict


def get_target_layer(model, name):
    """Derniere couche de convolution de chaque architecture."""
    if name == "resnet18":
        return model.layer4[-1]
    return model.features[-1]  # efficientnet_v2_s et convnext_tiny


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=1):
        """Carte (224x224, valeurs 0 a 1) pour la classe class_idx (1 = fissure)."""
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        proba = torch.softmax(logits.detach(), dim=1)[0, 1].item()
        return cam.cpu().numpy(), proba


def denormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "efficientnet_v2_s", "convnext_tiny"])
    p.add_argument("--name", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--surface", default=None, choices=[None, "D", "P", "W"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = get_loaders(batch_size=64)
    ds = val_loader.dataset

    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    y_true, y_score = predict(model, val_loader, device)
    pred = (y_score >= args.threshold).astype(int)
    surfaces = ds.df["surface"].values
    mask = np.ones(len(y_true), dtype=bool)
    if args.surface:
        mask = surfaces == args.surface

    cats = {
        "Fissure detectee (VP)": (y_true == 1) & (pred == 1),
        "Fissure ratee (FN)": (y_true == 1) & (pred == 0),
        "Fausse alerte (FP)": (y_true == 0) & (pred == 1),
        "Sain correct (VN)": (y_true == 0) & (pred == 0),
    }

    gc = GradCAM(model, get_target_layer(model, args.model))
    rng = random.Random(args.seed)
    n = args.n
    fig, axes = plt.subplots(4, 2 * n, figsize=(2.3 * 2 * n, 2.5 * 4))

    for r, (cat, cond) in enumerate(cats.items()):
        idx = np.where(cond & mask)[0].tolist()
        chosen = rng.sample(idx, min(n, len(idx)))
        print(f"{cat} : {len(idx)} images, {len(chosen)} montrees")
        for j in range(n):
            ax_img, ax_cam = axes[r][2 * j], axes[r][2 * j + 1]
            for ax in (ax_img, ax_cam):
                ax.set_xticks([])
                ax.set_yticks([])
            if j >= len(chosen):
                ax_img.axis("off")
                ax_cam.axis("off")
                continue
            i = chosen[j]
            x, _ = ds[i]
            cam, proba = gc(x.unsqueeze(0).to(device))
            img = denormalize(x)
            ax_img.imshow(img)
            ax_cam.imshow(img)
            ax_cam.imshow(cam, cmap="jet", alpha=0.45)
            ax_cam.set_title(f"p={proba:.2f} {surfaces[i]}", fontsize=8)
            print("   ", ds.df.iloc[i]["path"], f"p={proba:.3f}")
        axes[r][0].set_ylabel(cat, fontsize=8)

    fig.suptitle(f"Grad-CAM : {args.name} (seuil {args.threshold})", fontsize=11)
    fig.tight_layout()
    out = Path("reports/figures") / f"gradcam_{args.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Figure ecrite :", out)


if __name__ == "__main__":
    main()