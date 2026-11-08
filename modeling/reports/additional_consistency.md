# Consistencia en conjuntos adicionales (PI2 y PI3)

Modelo: `final.joblib`. Se excluyen todos los videos del ground truth. *Inapropiado* = `disturbing` o `restricted`. Sin etiquetas humanas: los valores miden consistencia con el clasificador original, no eficacia.

| Conjunto | Videos | % inapropiado (modelo) | % inapropiado (original) | Acuerdo % | Kappa | Ambos | Solo modelo | Solo original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elsagate_related | 230,873 | 21.78 | 0.55 | 77.87 | -0.002 | 227 | 50,051 | 1,033 |
| other_child_related | 137,043 | 11.92 | 0.43 | 87.67 | -0.007 | 8 | 16,323 | 578 |
| random | 462,382 | 6.74 | 0.73 | 92.62 | 0.000 | 231 | 30,945 | 3,157 |
| popular | 9,607 | 12.00 | 0.00 | 88.00 | 0.000 | 0 | 1,153 | 0 |

Distribución de las 4 clases predichas por el modelo (%):

| Conjunto | disturbing | restricted | suitable | irrelevant |
|---|---:|---:|---:|---:|
| elsagate_related | 8.59 | 13.19 | 15.21 | 63.01 |
| other_child_related | 4.86 | 7.05 | 26.31 | 61.77 |
| random | 1.20 | 5.55 | 3.56 | 89.70 |
| popular | 4.10 | 7.90 | 3.57 | 84.43 |

Las columnas *Solo modelo* y *Solo original* son los desacuerdos que se revisarán de forma cualitativa (H2).
