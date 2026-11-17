import csv
import pandas as pd

# Pour chaque modele : nom affiche, fichier d'historique
RUNS = {
    "EfficientNet-V2-S": "reports/metrics/efficientnet-v2-s_epochs.csv",
    "ConvNeXt-Tiny": "reports/metrics/convnext-tiny_epochs.csv",
}

lignes = []

# ResNet-18 baseline : entrainee avant le CSV, on reprend ses valeurs (JSON)
lignes.append({
    "Modele": "ResNet-18 (baseline)", "Meilleure epoque": 3,
    "PR-AUC": 0.882, "Recall": 0.713, "Precision": 0.892, "F1": 0.792,
    "TP": 995, "FN": 401, "FP": 121, "TN": 7012,
})

for nom, chemin in RUNS.items():
    df = pd.read_csv(chemin)
    b = df.loc[df["val_pr_auc"].idxmax()]  # meilleure epoque = meilleure PR-AUC
    lignes.append({
        "Modele": nom, "Meilleure epoque": int(b["epoch"]),
        "PR-AUC": round(b["val_pr_auc"], 3), "Recall": round(b["val_recall"], 3),
        "Precision": round(b["val_precision"], 3), "F1": round(b["val_f1"], 3),
        "TP": int(b["tp"]), "FN": int(b["fn"]), "FP": int(b["fp"]), "TN": int(b["tn"]),
    })

meilleur = max(lignes, key=lambda r: r["PR-AUC"])
champs = list(lignes[0].keys())

with open("reports/metrics/phase3_architectures.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["Meilleur modele (PR-AUC validation)", meilleur["Modele"], meilleur["PR-AUC"]])
    w.writerow([])
    w.writerow(champs)
    for r in lignes:
        w.writerow([r[c] for c in champs])

print(pd.DataFrame(lignes).to_string(index=False))
print("\nMeilleur modele :", meilleur["Modele"], "-> PR-AUC", meilleur["PR-AUC"])