import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RAW = ROOT / "data/raw/SDNET2018_images"
IDX = ROOT / "data/processed/index.csv"
OUT = ROOT / "reports/quality"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IDX)
if "exclure" in df.columns:          # le script peut être relancé sans risque
    df = df.drop(columns="exclure")

# 1. Regrouper les images par empreinte MD5 (même contenu = même empreinte)
groups = defaultdict(list)
for n, (i, path) in enumerate(df["path"].items(), 1):
    groups[hashlib.md5((RAW / path).read_bytes()).hexdigest()].append(i)
    if n % 10000 == 0:
        print(f"  {n}/{len(df)} images lues", flush=True)

# 2. Décider quelles images exclure
df["exclure"] = 0
rows = []
for h, idxs in groups.items():
    if len(idxs) < 2:
        continue
    sub = df.loc[idxs].sort_values("path")
    if sub["label"].nunique() > 1:
        a_exclure, motif = list(sub.index), "etiquettes contradictoires"
    else:
        a_exclure, motif = list(sub.index[1:]), "doublon exact"
    df.loc[a_exclure, "exclure"] = 1
    for j in a_exclure:
        rows.append({"groupe": h[:8], "path": df.at[j, "path"],
                     "label": df.at[j, "label"], "split": df.at[j, "split"],
                     "motif": motif})

# 3. Contrôles automatiques
assert df["exclure"].sum() == len(rows)
assert len(df) == 56092, "le nombre d'images a changé"
assert df.groupby("photo")["exclure"].min().max() == 0, "une photo entière est exclue"

# 4. Enregistrer l'index mis à jour et la liste lisible des exclusions
df.to_csv(IDX, index=False)
rep = pd.DataFrame(rows).sort_values(["groupe", "path"])
rep.to_csv(OUT / "exclusions.csv", index=False)

print()
print(f"Groupes de doublons : {sum(len(v) > 1 for v in groups.values())}")
print(f"Images exclues : {int(df['exclure'].sum())} sur {len(df)}")
print(rep["motif"].value_counts().to_string())
print(rep["split"].value_counts().to_string())
print("=== ETAPE 2.1a OK ✅ ===")