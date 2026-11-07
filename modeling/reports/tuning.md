# Ajuste de hiperparámetros (v1)

Método: `RandomizedSearchCV` con los 5 folds agrupados por canal (`folds_v1.csv`), solo sobre train. Criterio de selección: F1 macro medio en CV.

| Modelo | Espacio de búsqueda | Combinaciones | Mejores hiperparámetros | F1 macro CV | Recall restricted CV |
|---|---|---:|---|---:|---:|
| LogReg combinado (TF-IDF + metadatos) | `C` ∈ [0.05, 20] (log-uniforme) | 20 | `C=1.16` | 0.710 ± 0.024 | 0.558 |
| LightGBM combinado (SVD 200 + metadatos) | `n_estimators` ∈ [200, 1000]; `learning_rate` ∈ [0.01, 0.2] (log); `num_leaves` ∈ [8, 64]; `min_child_samples` ∈ [5, 60]; `colsample_bytree` ∈ [0.4, 1.0]; `reg_lambda` ∈ [0.001, 10] (log) | 20 | `colsample_bytree=0.4444`, `learning_rate=0.02927`, `min_child_samples=45`, `n_estimators=227`, `num_leaves=14`, `reg_lambda=2.513` | 0.698 ± 0.027 | 0.478 |

**Modelo seleccionado:** LogReg combinado (TF-IDF + metadatos).
