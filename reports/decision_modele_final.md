
# Décision : modèle final (étape 3.4)

Date : 24 novembre 2026. **Le jeu de test n'a pas encore été utilisé.**

## Règle de choix (fixée avant de regarder les résultats)

1. Critère principal : la plus haute précision à recall >= 0,85 sur la validation.
2. Si l'écart est inférieur à 0,02 : le meilleur recall sur les ponts.
3. Ensuite : la vitesse d'inférence.

Le seuil de décision est choisi sur la validation. Le test n'est mesuré qu'une seule fois,
après cette décision.

## Résultats sur la validation (recall >= 0,85)

| Modèle           | Seuil | Recall | Précision | F1    |
| ----------------- | ----- | ------ | ---------- | ----- |
| ConvNeXt-Tiny     | 0,05  | 0,867  | 0,707      | 0,779 |
| EfficientNet-V2-S | 0,20  | 0,862  | 0,663      | 0,749 |

Écart de précision : +0,044 pour ConvNeXt-Tiny (supérieur à 0,02).

Par surface (recall / précision) :

| Surface | EfficientNet-V2-S | ConvNeXt-Tiny |
| ------- | ----------------- | ------------- |
| Ponts   | 0,806 / 0,622     | 0,783 / 0,642 |
| Routes  | 0,832 / 0,707     | 0,858 / 0,725 |
| Murs    | 0,903 / 0,660     | 0,910 / 0,727 |

## Décision

**Modèle final : ConvNeXt-Tiny, seuil 0,05** (`models/convnext-tiny.pt`).

## Limites connues

- Chaque modèle n'a été entraîné qu'une fois : l'écart reste à confirmer avec plusieurs graines.
- Le seuil 0,05 est sensible : les probabilités de ConvNeXt-Tiny sont concentrées près de 0
  (recall 0,837 au seuil 0,07 et 0,867 au seuil 0,05).
- Les ponts restent la surface la plus difficile (recall 0,783).
- EfficientNet-V2-S donne un meilleur recall sur les ponts (0,806 contre 0,783).
