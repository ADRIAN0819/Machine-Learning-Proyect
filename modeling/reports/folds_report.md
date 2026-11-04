# Reporte de la partición agrupada por canal (v1)

Generado por `modeling/splits.py` con `StratifiedGroupKFold` (5 folds, semilla 42), agrupando por `channel_id`.
El test es la columna `split` de `dataset_features_v1.csv`; los folds se calculan solo dentro de train.

| Partición | Videos | Canales | % disturbing | % irrelevant | % restricted | % suitable |
|---|---:|---:|---:|---:|---:|---:|
| fold 0 | 745 | 608 | 17.6 | 40.5 | 8.9 | 33.0 |
| fold 1 | 747 | 608 | 19.3 | 42.6 | 11.0 | 27.2 |
| fold 2 | 784 | 614 | 22.2 | 41.7 | 6.8 | 29.3 |
| fold 3 | 768 | 614 | 19.7 | 38.0 | 8.9 | 33.5 |
| fold 4 | 798 | 608 | 17.2 | 41.9 | 8.8 | 32.2 |
| test | 955 | 766 | 20.1 | 38.0 | 8.4 | 33.5 |
| **ground truth** | 4,797 | 3,818 | 19.4 | 40.4 | 8.7 | 31.5 |

- Desviación máxima de la proporción de una clase respecto al total: 4.4 puntos porcentuales.
- Ningún canal aparece en más de una partición (verificado por `_check_folds`).
- Nota: las imputaciones de `dataset_features_v1.csv` se ajustaron con todo train; el efecto dentro de la validación cruzada es mínimo (medianas), pero las transformaciones que aprenden de los datos (TF-IDF, escalado, SMOTE) deben ir dentro de un `Pipeline` para ajustarse solo con los folds de entrenamiento.
