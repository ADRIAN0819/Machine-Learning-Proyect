# Modelo final en el conjunto de prueba (v1)

Modelo: **LogReg combinado (TF-IDF + metadatos)**, con `C=1.16`.
Elegido por F1 macro en validación cruzada (ver `tuning.md`), reentrenado con todo train (3,842 videos) y evaluado una sola vez en test (955 videos de canales no vistos en entrenamiento).

| Métrica | Valor |
|---|---:|
| F1 macro | 0.725 |
| Recall macro | 0.724 |
| Accuracy | 0.775 |

| Clase | Precision | Recall | F1 | Videos |
|---|---:|---:|---:|---:|
| disturbing | 0.801 | 0.714 | 0.755 | 192 |
| irrelevant | 0.794 | 0.837 | 0.815 | 363 |
| restricted | 0.500 | 0.550 | 0.524 | 80 |
| suitable | 0.815 | 0.797 | 0.806 | 320 |

Matriz de confusión (filas = clase real, columnas = predicción):

| Real \ Predicho | disturbing | irrelevant | restricted | suitable |
|---|---:|---:|---:|---:|
| disturbing | 137 | 17 | 9 | 29 |
| irrelevant | 16 | 304 | 24 | 19 |
| restricted | 4 | 22 | 44 | 10 |
| suitable | 14 | 40 | 11 | 255 |
