import hashlib, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RAW = ROOT / "data/raw/SDNET2018_images"
IDX = ROOT / "data/processed/index.csv"
OUT = ROOT / "reports/quality"   # hors de data/processed (suivi par DVC)
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(IDX)

corrupted, sizes, modes = [], Counter(), Counter()
by_md5 = defaultdict(list)
for i, row in enumerate(df.itertuples(index=False), 1):
    p = RAW / row.path
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    except Exception as e:
        corrupted.append((row.path, type(e).__name__))
        continue
    by_md5[hashlib.md5(p.read_bytes()).hexdigest()].append(row)
    if i % 10000 == 0:
        print(f"  {i}/{len(df)} images controlees", flush=True)

dups = {h: r for h, r in by_md5.items() if len(r) > 1}
extra = sum(len(r) - 1 for r in dups.values())
cross_label = sum(1 for r in dups.values() if len({x.label for x in r}) > 1)
cross_split = sum(1 for r in dups.values() if len({x.split for x in r}) > 1)

L = ["=== QUALITE DES DONNEES SDNET2018 ===",
     f"Images dans l'index : {len(df)}",
     f"Images corrompues : {len(corrupted)}"]
L += [f"  - {p} ({e})" for p, e in corrupted[:20]]
L += ["", "Dimensions (largeur, hauteur) :"] + [f"  {k} : {v}" for k, v in sizes.most_common()]
L += ["Modes couleur : " + ", ".join(f"{k}={v}" for k, v in modes.most_common())]
L += ["", f"Groupes de doublons exacts (meme contenu) : {len(dups)}",
      f"Fichiers en trop (a supprimer si on dedoublonne) : {extra}",
      f"Groupes avec des etiquettes differentes : {cross_label}",
      f"Groupes repartis sur plusieurs splits (fuite) : {cross_split}"]
for h, r in list(dups.items())[:10]:
    L.append("  " + " | ".join(f"{x.path}[{x.split},{x.label}]" for x in r[:4]))
report = "\n".join(L)
(OUT / "quality_report.txt").write_text(report, encoding="utf-8")
print(report)

# planche d'exemples : 4 patchs fissure + 4 sains par surface (train, graine 42)
rng = np.random.default_rng(42)
bad = {p for p, _ in corrupted}
tr = df[(df.split == "train") & (~df.path.isin(bad))]
names = {"D": "Pont (D)", "P": "Chaussee (P)", "W": "Mur (W)"}
fig, axes = plt.subplots(6, 4, figsize=(8, 12))
r = 0
for s in ("D", "P", "W"):
    for lab, txt in ((1, "fissure"), (0, "sans fissure")):
        sub = tr[(tr.surface == s) & (tr.label == lab)]
        pick = sub.iloc[rng.choice(len(sub), 4, replace=False)]
        for c, row in enumerate(pick.itertuples(index=False)):
            ax = axes[r, c]; ax.imshow(Image.open(RAW / row.path)); ax.axis("off")
            if c == 0:
                ax.set_title(f"{names[s]} - {txt}", fontsize=9, loc="left")
        r += 1
fig.tight_layout(); fig.savefig(OUT / "sample_grid.png", dpi=130)
print("Planche d'exemples : reports/quality/sample_grid.png")
print("=== ETAPE 0.1 OK ✅ ===")