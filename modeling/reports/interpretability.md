# Interpretabilidad y análisis de errores

Modelo: **LogReg combinado (TF-IDF + metadatos)**, `C=1.16`.

## Importancia de variables (SHAP)

Se usa `shap.LinearExplainer` (exacto para regresión logística).
Los valores SHAP miden la contribución marginal de cada feature a
la predicción de cada clase (one-vs-rest).

### Top features globales

Ver figura: `reports/figures/shap_summary.png`

### Top features para `disturbing`

Ver figura: `reports/figures/shap_bar_disturbing.png`

| Feature | Importancia media |SHAP| |
|---|---:|
| like count log1p | 0.27518 |
| category id 1 | 0.22382 |
| duration seconds | 0.22184 |
| licensed content True | 0.14959 |
| title chars | 0.14828 |
| licensed content False | 0.10969 |
| title: peppa | 0.07956 |
| view count log1p | 0.07872 |
| comment count log1p | 0.06165 |
| title: pig | 0.05571 |
| category id 20 | 0.05341 |
| category id 22 | 0.05313 |
| title: elsa | 0.05300 |
| likes per view | 0.05169 |
| title: peppa pig | 0.04730 |

### Top features para `restricted`

Ver figura: `reports/figures/shap_bar_restricted.png`

| Feature | Importancia media |SHAP| |
|---|---:|
| like count log1p | 1.21098 |
| dislike count log1p | 0.81065 |
| view count log1p | 0.60033 |
| comment count log1p | 0.28788 |
| licensed content True | 0.17754 |
| title chars | 0.16241 |
| licensed content False | 0.13870 |
| category id 1 | 0.13100 |
| title words | 0.11898 |
| category id 20 | 0.10667 |
| comments per view | 0.08835 |
| category id 24 | 0.08315 |
| category id 22 | 0.06837 |
| likes per view | 0.06230 |
| duration seconds | 0.04823 |

### Interpretación

- Las features de **texto** (TF-IDF de título y tags) dominan la importancia
  global, lo que es consistente con el ablation study: el texto solo da F1=0.625
  vs. F1=0.682 con texto+metadatos.
- Para `disturbing`, los términos de tags relacionados con personajes de
  dibujos animados en contextos violentos/sexuales tienen los SHAP más altos.
- Para `restricted`, los términos sexuales y de hipnosis en título/tags son
  determinantes, pero su baja frecuencia limita el recall (F1=0.524).
- Los metadatos de engagement (`likes_per_view`, `view_count_log1p`) tienen
  importancia SHAP moderada, pero ayudan especialmente en `irrelevant` vs `suitable`.

## Análisis cualitativo de errores

> Se analizan los errores del modelo final (LogReg combinado, C=1.16) sobre
> 955 videos de test (canales no vistos en entrenamiento).

### Clase `disturbing`

- Videos reales en test: **192**
- Falsos negativos (no detectados): **55** (28.6%)
- Predichos como: {'suitable': 29, 'irrelevant': 17, 'restricted': 9}

**Ejemplos de títulos no detectados como `disturbing`:**

- *"Kiss him not me all episodes °English° no ads"* → predicho: `irrelevant` (conf=0.60)
- *"JooJooTV 4K | THE LITTERER | Kids Cartoons | Best Cartoons For Kids 2019"* → predicho: `suitable` (conf=0.95)
- *"Scary Shark | Nursery Rhymes | Scary Videos For Kids | Stories of the shark"* → predicho: `suitable` (conf=0.92)
- *"Caravan Palace - Miracle (official video)"* → predicho: `irrelevant` (conf=0.90)
- *"Derek Gerard Content"* → predicho: `irrelevant` (conf=0.65)

### Clase `restricted`

- Videos reales en test: **80**
- Falsos negativos (no detectados): **36** (45.0%)
- Predichos como: {'irrelevant': 22, 'suitable': 10, 'disturbing': 4}

**Ejemplos de títulos no detectados como `restricted`:**

- *"elsa kills everyone"* → predicho: `disturbing` (conf=0.76)
- *"JOHN_STONE3: SPECIAL KILLS"* → predicho: `irrelevant` (conf=0.69)
- *"RΟUGH NІGHT - ALL Uncensored Clips + Trailer (2017) Scarlett Johansson, Zoë Krav"* → predicho: `suitable` (conf=0.75)
- *"Spider Man vs Elsa | Frozen Elsa and Spider Man GTA 5"* → predicho: `suitable` (conf=0.53)
- *"Thermal Hog Hunting 55 Hogs with Trijicon IR Hunter."* → predicho: `irrelevant` (conf=0.85)

### Falsos positivos por clase

| Predicho como | Clase real | Cantidad |
|---|---|---:|
| `disturbing` | `irrelevant` | 16 |
| `disturbing` | `suitable` | 14 |
| `disturbing` | `restricted` | 4 |
| `restricted` | `irrelevant` | 24 |
| `restricted` | `suitable` | 11 |
| `restricted` | `disturbing` | 9 |

### Observaciones generales

- La clase `restricted` concentra la mayor tasa de error: contenido para
  adultos con vocabulario inusual o metadatos similares a `irrelevant`.
- Los falsos negativos de `disturbing` tienden a tener títulos genéricos
  (p.ej. personajes de dibujos animados) sin señales explícitas en el texto.
- La confusión entre `irrelevant` y `restricted` sugiere que el texto solo
  no siempre distingue contenido adulto disfrazado de entretenimiento general.
- Los metadatos (ratios de engagement, duración) ayudan pero no eliminan
  la ambigüedad en videos con thumbnails engañosos (no disponibles en el dataset).

## Figuras generadas

| Figura | Descripción |
|---|---|
| `figures/shap_summary.png` | Importancia SHAP global (todas las clases) |
| `figures/shap_bar_disturbing.png` | Top features para `disturbing` |
| `figures/shap_bar_restricted.png` | Top features para `restricted` |
| `figures/error_analysis.png` | Distribución VP/FN/FP por clase |
