# Interpretabilidad y análisis de errores

Modelo: **LogReg combinado (TF-IDF + metadatos)**, `C=1.16`.

## Importancia de variables (SHAP)

Se usa `shap.LinearExplainer` (exacto para regresión logística), con train
como fondo de referencia y los 955 videos de test como muestras explicadas.
Los valores SHAP miden la contribución marginal de cada feature a
la predicción de cada clase (one-vs-rest).

### Top features globales

Ver figura: `reports/figures/shap_summary.png`

### Top features para `disturbing`

Ver figura: `reports/figures/shap_bar_disturbing.png`

| Feature | Importancia media |SHAP| |
|---|---:|
| like count log1p | 0.27179 |
| duration seconds | 0.24738 |
| category id 1 | 0.21920 |
| licensed content True | 0.15078 |
| title chars | 0.14877 |
| licensed content False | 0.10849 |
| view count log1p | 0.07916 |
| title: peppa | 0.06893 |
| category id 20 | 0.06637 |
| comment count log1p | 0.05991 |
| title: elsa | 0.05455 |
| likes per view | 0.05298 |
| category id 22 | 0.05289 |
| title: pig | 0.05033 |
| category id 27 | 0.04431 |

### Top features para `restricted`

Ver figura: `reports/figures/shap_bar_restricted.png`

| Feature | Importancia media |SHAP| |
|---|---:|
| like count log1p | 1.19389 |
| dislike count log1p | 0.80894 |
| view count log1p | 0.58931 |
| comment count log1p | 0.28370 |
| licensed content True | 0.17741 |
| title chars | 0.16411 |
| licensed content False | 0.13664 |
| category id 20 | 0.13059 |
| category id 1 | 0.12731 |
| title words | 0.11897 |
| comments per view | 0.09460 |
| category id 24 | 0.08120 |
| category id 22 | 0.06805 |
| likes per view | 0.06447 |
| duration seconds | 0.05373 |

### Interpretación

- De forma individual, las features más influyentes son los conteos de
  interacción (likes, vistas, dislikes, comentarios), la categoría y
  `licensed_content`; los primeros términos de texto (peppa, pig, elsa) aparecen
  más abajo en el ranking.
- La importancia del texto se reparte entre ~38 mil columnas TF-IDF. Sumada por
  bloque, el texto aporta el 65 % de la importancia en `disturbing` y el 55 % en
  `suitable`; en `restricted` e `irrelevant` pesan más los metadatos (~60 %).
- Es consistente con el ablation study: texto solo F1=0.625, metadatos solos
  F1=0.580, texto+metadatos F1=0.682 (LightGBM).

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
