import csv
import json
from pathlib import Path

DIR = Path("reports/metrics")
MODELES = {
    "Baseline": "resnet18-baseline",
    "Poids 3": "resnet18-poids3",
    "Equilibre": "resnet18-balanced",
}
CIBLES = [0.80, 0.83, 0.88]


def lire_seuils(nom):
    with open(DIR / f"{nom}_seuils.csv", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def lire_pr_auc(nom):
    data = json.loads((DIR / f"{nom}_val.json").read_text(encoding="utf-8"))
    return data["global"]["pr_auc"]


def main():
    noms = list(MODELES.keys())
    seuils = {m: lire_seuils(MODELES[m]) for m in noms}
    pr_auc = {m: lire_pr_auc(MODELES[m]) for m in noms}
    meilleur_f1 = {m: max(seuils[m], key=lambda r: r["f1"]) for m in noms}

    out = DIR / "synthese.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")

        # Tableau 1 : le meilleur modele
        w.writerow(["Meilleur modele"] + noms)
        w.writerow(["PR-AUC"] + [f"{pr_auc[m]:.3f}" for m in noms])
        w.writerow(["Meilleur F1 (seuil)"] + [
            f"{meilleur_f1[m]['f1']:.3f} (seuil {meilleur_f1[m]['seuil']})" for m in noms
        ])
        w.writerow([])

        # Tableau 2 : precision a recall egal
        w.writerow(["Recall vise"] + [f"{m} (seuil)" for m in noms])
        for cible in CIBLES:
            choix = {m: min(seuils[m], key=lambda r: abs(r["recall"] - cible)) for m in noms}
            meilleure = max(choix[m]["precision"] for m in noms)
            ligne = [f"~{cible:.2f}"]
            for m in noms:
                r = choix[m]
                txt = f"{r['precision']:.3f} ({r['seuil']})"
                if abs(r["recall"] - cible) > 0.005:
                    txt += f", recall {r['recall']:.3f}"
                if r["precision"] == meilleure:
                    txt += " *"
                ligne.append(txt)
            w.writerow(ligne)
        w.writerow([])
        w.writerow(["* = meilleure precision a recall egal (ligne par ligne)"])

    print("Fichier ecrit :", out)


if __name__ == "__main__":
    main()