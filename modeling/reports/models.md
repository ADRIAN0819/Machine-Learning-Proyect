# Modelos tabulares, combinado y ablation study (v1)

Validación cruzada de 5 folds agrupada por canal (`folds_v1.csv`), solo sobre train (3,842 videos). Media ± desviación estándar entre folds. Hiperparámetros iniciales, todavía sin ajustar.

## Modelos comparados

| Modelo | F1 macro | Recall macro | Recall disturbing | Recall restricted | F1 disturbing | F1 irrelevant | F1 restricted | F1 suitable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest (metadatos) | 0.575 ± 0.027 | 0.564 ± 0.024 | 0.457 ± 0.041 | 0.323 ± 0.040 | 0.516 | 0.750 | 0.370 | 0.663 |
| LightGBM (metadatos) | 0.580 ± 0.020 | 0.573 ± 0.018 | 0.496 ± 0.044 | 0.326 ± 0.056 | 0.512 | 0.752 | 0.371 | 0.684 |
| LightGBM combinado (SVD 200 + metadatos) | 0.682 ± 0.020 | 0.662 ± 0.018 | 0.621 ± 0.059 | 0.331 ± 0.047 | 0.682 | 0.819 | 0.426 | 0.799 |
| LogReg combinado (TF-IDF + metadatos) | 0.710 ± 0.023 | 0.713 ± 0.021 | 0.664 ± 0.043 | 0.566 ± 0.042 | 0.697 | 0.820 | 0.528 | 0.796 |

## Ablation study (LightGBM)

Mismo modelo y mismos hiperparámetros; solo cambia la información de entrada.

| Entrada | F1 macro | Recall macro | Recall disturbing | Recall restricted | F1 disturbing | F1 irrelevant | F1 restricted | F1 suitable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Solo texto (SVD 200) | 0.625 ± 0.002 | 0.608 ± 0.005 | 0.609 ± 0.051 | 0.157 ± 0.041 | 0.687 | 0.782 | 0.250 | 0.782 |
| Solo metadatos | 0.580 ± 0.020 | 0.573 ± 0.018 | 0.496 ± 0.044 | 0.326 ± 0.056 | 0.512 | 0.752 | 0.371 | 0.684 |
| Texto + metadatos | 0.682 ± 0.020 | 0.662 ± 0.018 | 0.621 ± 0.059 | 0.331 ± 0.047 | 0.682 | 0.819 | 0.426 | 0.799 |
