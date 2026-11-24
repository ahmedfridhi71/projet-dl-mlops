import argparse
import csv
import json
from pathlib import Path

MODELES = [
    ("ResNet-18", "resnet18-baseline"),
    ("EfficientNet-V2-S", "efficientnet-v2-s"),
    ("ConvNeXt-Tiny", "convnext-tiny"),
]
NOMS = {"D": "Ponts (D)", "P": "Routes (P)", "W": "Murs (W)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="r80",
                   help="suffixe des JSON : r80 (recall ~0.80) ou r85 (recall >= 0.85)")
    args = p.parse_args()

    lignes = []
    for nom, base in MODELES:
        chemin = Path(f"reports/metrics/{base}-{args.suffix}_val.json")
        if not chemin.exists():
            print("Ignore (fichier absent) :", chemin)
            continue
        d = json.loads(chemin.read_text(encoding="utf-8"))
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
    nom_sortie = "phase3_par_surface" if args.suffix == "r80" else f"phase3_par_surface_{args.suffix}"
    out = Path(f"reports/metrics/{nom_sortie}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(champs)
        for r in lignes:
            w.writerow([r[c] for c in champs])

    for r in lignes:
        print(r)
    print("Fichier ecrit :", out)


if __name__ == "__main__":
    main()