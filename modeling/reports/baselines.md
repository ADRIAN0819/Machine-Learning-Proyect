# Modelos baseline (v1)

Validación cruzada de 5 folds agrupada por canal (`folds_v1.csv`), solo sobre train (3,842 videos). Media ± desviación estándar entre folds.

| Modelo | F1 macro | Recall macro | Recall disturbing | Recall restricted | Accuracy |
|---|---:|---:|---:|---:|---:|
| Dummy (clase mayoritaria) | 0.145 ± 0.004 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.409 ± 0.016 |
| LogReg + TF-IDF | 0.672 ± 0.025 | 0.656 ± 0.021 | 0.637 ± 0.048 | 0.359 ± 0.073 | 0.745 ± 0.019 |
| Naive Bayes Multinomial + TF-IDF | 0.659 ± 0.033 | 0.639 ± 0.031 | 0.670 ± 0.055 | 0.254 ± 0.057 | 0.746 ± 0.026 |

F1 por clase:

| Modelo | disturbing | irrelevant | restricted | suitable |
|---|---:|---:|---:|---:|
| Dummy (clase mayoritaria) | 0.000 | 0.581 | 0.000 | 0.000 |
| LogReg + TF-IDF | 0.690 | 0.791 | 0.422 | 0.786 |
| Naive Bayes Multinomial + TF-IDF | 0.703 | 0.793 | 0.360 | 0.779 |

- No se reporta accuracy como métrica principal: con 40 % de `irrelevant`, el modelo trivial ya obtiene una accuracy alta sin detectar ningún video `disturbing` ni `restricted`.
