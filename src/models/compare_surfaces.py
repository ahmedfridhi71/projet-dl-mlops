import csv
import json
from pathlib import Path

MODELES = [
    ("ResNet-18", "resnet18-baseline-r80"),
    ("EfficientNet-V2-S", "efficientnet-v2-s-r80"),
    ("ConvNeXt-Tiny", "convnext-tiny-r80"),
]
NOMS = {"D": "Ponts (D)", "P": "Routes (P)", "W": "Murs (W)"}

lignes = []
for nom, run in MODELES:
    d = json.loads(Path(f"reports/metrics/{run}_val.json").read_text(encoding="utf-8"))
    g = d["global"]
    ligne = {
        "Modele": nom,
        "Seuil": d["threshold"],
        "Recall global": round(g["recall"], 3),
        "Precision globale": round(g["precision"], 3),
    }
    for s, m in d["by_surface"].items():
        ligne[f"Recall {NOMS.get(s, s)}"] = round(m["recall"], 3)
    for s, m in d["by_surface"].items():
        ligne[f"Precision {NOMS.get(s, s)}"] = round(m["precision"], 3)
    lignes.append(ligne)

champs = list(lignes[0].keys())
out = Path("reports/metrics/phase3_par_surface.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(champs)
    for r in lignes:
        w.writerow([r[c] for c in champs])

for r in lignes:
    print(r)
print("Fichier ecrit :", out)