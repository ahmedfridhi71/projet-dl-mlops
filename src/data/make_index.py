from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
RAW = Path("data/raw/SDNET2018_images")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SURFACES = {"D": ("CD", "UD"), "P": ("CP", "UP"), "W": ("CW", "UW")}

rows = []
for s, (crack, clean) in SURFACES.items():
    for folder, label in ((crack, 1), (clean, 0)):
        for p in sorted((RAW / s / folder).glob("*.jpg")):
            photo = p.stem.split("-")[0]
            rows.append({
                "path": p.relative_to(RAW).as_posix(),
                "label": label,
                "surface": s,
                "photo": f"{s}_{photo}",
            })
df = pd.DataFrame(rows)

# Split par photo, separement pour chaque surface
rng = np.random.default_rng(SEED)
split_of = {}
for s in SURFACES:
    photos = sorted(df.loc[df.surface == s, "photo"].unique())
    rng.shuffle(photos)
    n = len(photos)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    for i, ph in enumerate(photos):
        split_of[ph] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
df["split"] = df["photo"].map(split_of)

# Controles
assert len(df) == 56092, f"Total inattendu : {len(df)}"
assert df["split"].notna().all(), "Patchs sans split"
assert df.groupby("photo")["split"].nunique().max() == 1, "FUITE : une photo est dans plusieurs splits"

df.to_csv(OUT / "index.csv", index=False)

lines = ["=== PATCHS PAR SPLIT ==="]
g = df.groupby("split").agg(patchs=("path", "size"),
                            fissures=("label", "sum"),
                            photos=("photo", "nunique"))
g["taux_fissure_%"] = (100 * g["fissures"] / g["patchs"]).round(1)
lines.append(g.to_string())
lines.append("")
lines.append("=== PAR SPLIT ET SURFACE ===")
g2 = df.groupby(["split", "surface"]).agg(patchs=("path", "size"),
                                          fissures=("label", "sum"),
                                          photos=("photo", "nunique"))
g2["taux_fissure_%"] = (100 * g2["fissures"] / g2["patchs"]).round(1)
lines.append(g2.to_string())
report = "\n".join(lines)
(OUT / "split_report.txt").write_text(report, encoding="utf-8")
print(report)
print()
print("Aucune photo en commun entre les splits : OK")
print("=== ETAPE 1.3b OK ✅ ===")
