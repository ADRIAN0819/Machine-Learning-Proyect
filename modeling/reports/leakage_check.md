# Fuga de datos por canal: validación aleatoria vs. agrupada

Modelo: TF-IDF (1-2 gramas) sobre título + tags + descripción y regresión logística con `class_weight='balanced'`. 5 folds sobre train (3,842 videos).

| Validación | F1 macro | Recall macro | Accuracy |
|---|---:|---:|---:|
| Aleatorio (StratifiedKFold) | 0.699 ± 0.020 | 0.681 ± 0.020 | 0.766 ± 0.014 |
| Agrupado por canal (folds_v1) | 0.678 ± 0.028 | 0.661 ± 0.025 | 0.751 ± 0.018 |

Diferencia en F1 macro (aleatorio − agrupado): **+0.020**. Una diferencia positiva indica que la validación aleatoria sobreestima el desempeño porque el modelo reconoce canales vistos en entrenamiento.
