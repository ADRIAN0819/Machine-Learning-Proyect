# Evaluación del modelo final en el conjunto de prueba

Modelo: **LogReg combinado (TF-IDF + metadatos)**, `C=1.16`.
Evaluado una sola vez sobre 955 videos de canales no vistos en entrenamiento.

## Métricas globales

| Métrica | Valor |
|---|---:|
| F1 macro | 0.725 |
| Recall macro | 0.724 |
| Accuracy | 0.775 |
| AUC macro (ROC, OvR) | 0.914 |
| AP macro (Prec-Recall) | 0.772 |

## Métricas por clase

| Clase | Precision | Recall | F1 | AUC | AP | Videos |
|---|---:|---:|---:|---:|---:|---:|
| disturbing | 0.801 | 0.714 | 0.755 | 0.929 | 0.848 | 192 |
| irrelevant | 0.794 | 0.837 | 0.815 | 0.921 | 0.873 | 363 |
| restricted | 0.500 | 0.550 | 0.524 | 0.878 | 0.483 | 80 |
| suitable | 0.815 | 0.797 | 0.806 | 0.929 | 0.885 | 320 |

## Matriz de confusión

> Filas = clase real, columnas = predicción.

| Real \ Predicho | disturbing | irrelevant | restricted | suitable |
|---|---:|---:|---:|---:|
| disturbing | 137 | 17 | 9 | 29 |
| irrelevant | 16 | 304 | 24 | 19 |
| restricted | 4 | 22 | 44 | 10 |
| suitable | 14 | 40 | 11 | 255 |

## Figuras generadas

| Figura | Archivo |
|---|---|
| Curvas ROC | `reports/figures/roc_curves.png` |
| Curvas Precision-Recall | `reports/figures/pr_curves.png` |
| Matriz de confusión | `reports/figures/confusion_matrix.png` |

## Notas metodológicas

- Se usa **F1 macro** como métrica principal (no accuracy) porque el dataset
  está desbalanceado: `irrelevant` representa el 40 % del total.
  Con accuracy el modelo trivial (clase mayoritaria) ya obtendría ~41 %
  sin detectar ningún video `disturbing` ni `restricted`.
- Las curvas ROC y PR se calculan en modo **one-vs-rest** para cada clase.
- `restricted` tiene el menor soporte (80 videos en test) y el AUC/AP más bajo,
  lo que refleja la dificultad de distinguirla de `irrelevant`.
- `class_weight='balanced'` en el modelo compensa el desbalance de clases
  sin generar datos sintéticos (SMOTE), para evitar sobreajuste en train.
