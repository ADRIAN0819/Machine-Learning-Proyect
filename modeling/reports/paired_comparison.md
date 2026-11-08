# Comparación pareada: regresión logística vs. LightGBM (ajustados)

Mismos 5 folds agrupados por canal (`folds_v1.csv`), solo train. Hiperparámetros: los mejores de `RandomizedSearchCV` (ver `tuning.md`).

- LogReg combinado (TF-IDF + metadatos): `C=1.16`
- LightGBM combinado (SVD 200 + metadatos): `colsample_bytree=0.4444`, `learning_rate=0.02927`, `min_child_samples=45`, `n_estimators=227`, `num_leaves=14`, `reg_lambda=2.513`

## F1 macro por fold

| Fold | Regresión logística | LightGBM | Diferencia (RL - LGBM) |
|---:|---:|---:|---:|
| 0 | 0.675 | 0.663 | +0.012 |
| 1 | 0.744 | 0.730 | +0.014 |
| 2 | 0.692 | 0.670 | +0.021 |
| 3 | 0.721 | 0.707 | +0.014 |
| 4 | 0.718 | 0.719 | -0.001 |

## Resumen pareado

| Métrica | RL (media ± d.e.) | LightGBM (media ± d.e.) | Dif. media ± d.e. | Folds a favor de RL | t pareada | p |
|---|---:|---:|---:|---:|---:|---:|
| f1_macro | 0.710 ± 0.024 | 0.698 ± 0.027 | +0.012 ± 0.008 | 4/5 | 3.23 | 0.032 |
| recall_restricted | 0.558 ± 0.042 | 0.478 ± 0.063 | +0.080 ± 0.042 | 5/5 | 4.29 | 0.013 |
| recall_disturbing | 0.668 ± 0.044 | 0.643 ± 0.060 | +0.026 ± 0.041 | 4/5 | 1.41 | 0.231 |

## Lectura

- F1 macro: la regresión logística supera a LightGBM en 4 de 5 folds, con una diferencia media de +0.012 (p = 0.032).
- Recall de restricted: diferencia media de +0.080, a favor de la regresión logística en 5 de 5 folds (p = 0.013).
- Con 5 folds la prueba t tiene poca potencia y los folds no son del todo independientes (comparten entrenamiento), así que el p-valor es orientativo. La elección de la regresión logística se apoya además en su interpretabilidad y en que usa el TF-IDF completo.
