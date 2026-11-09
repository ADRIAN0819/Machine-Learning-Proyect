# Proyecto Machine Learning: Detección de Videos No Aptos para Niños en YouTube

Repositorio para la clasificación y detección de videos inapropiados o perturbadores dirigidos a niños en YouTube (fenómeno Elsagate), basado en metadatos, texto (NLP) y métricas de interacción.

---

## Integrantes

* **Mariela Ocampo** (UTEC)
* **María Lazón** (UTEC)
* **Leonardo Chocce** (UTEC)
* **Adrián Urbina** (UTEC)

---

## Objetivos del proyecto

**Objetivo general.** Desarrollar un modelo de aprendizaje supervisado que clasifique automáticamente videos de YouTube según su idoneidad para niños pequeños (`suitable`, `irrelevant`, `restricted`, `disturbing`) a partir de sus metadatos, como herramienta de apoyo a la moderación de contenido.

**Objetivos específicos.**
1. Caracterizar el dataset *Disturbed YouTube for Kids* mediante un análisis exploratorio: estructura, valores faltantes, outliers, desbalance de clases, relaciones entre variables y riesgos de fuga de datos. *(Etapa 1)*
2. Diseñar el preprocesamiento y la ingeniería de características a partir del texto y de las métricas de interacción. *(Etapa 2)*
3. Entrenar y comparar un baseline y al menos dos enfoques de modelado, con una validación que agrupe por canal. *(Etapa 2)*
4. Evaluar el desempeño con métricas apropiadas para clases desbalanceadas, con énfasis en el recall de `disturbing` y `restricted`, y analizar los errores. *(Etapa 2)*
5. Comprobar la consistencia de las predicciones del modelo sobre los conjuntos adicionales, que no tienen etiquetas humanas fuera del ground truth. *(Etapa 2)*

---

## Estructura del Repositorio

* `exploratory_data_analysis/`: Análisis exploratorio de datos (EDA), reportes técnicos y figuras.
  * `eda_analysis.ipynb`: Notebook principal de análisis exploratorio (figuras originales de alta resolución guardadas en `images/`, usadas en `EDA.md` y en la presentación).
  * `eda_analysis_overleaf.ipynb`: Notebook especializado para el paper (genera las figuras a tamaño final IEEE dos columnas en `images_overleaf/`, que se suben al proyecto de Overleaf).
  * `EDA.md`: Documento consolidado e informe técnico completo del análisis exploratorio.
  * `images/`: Gráficos generados por el notebook principal a escala completa (14 figuras originales).
  * `images_overleaf/`: Gráficos generados para el artículo científico a escala 1:1 IEEE (14 figuras).
* `data_processing/`: Preprocesamiento y dataset de características v1.
  * `preprocesamiento_features_v1.ipynb`: Notebook para auditar, transformar e imputar el ground truth.
  * `preprocessing.py`: Función reutilizable de ingeniería de características y exportación.
  * `dataset_features_v1.csv`: 16 features procesadas (incluye longitud del título), etiqueta y metadatos (`video_id`, `channel_id`, `split`) para la partición agrupada por canal.
* `modeling/`: Partición, modelos, ajuste, evaluación, interpretabilidad y comprobación de consistencia (Etapa 2).
  * `splits.py`: Folds de validación cruzada (5, estratificados y agrupados por canal) dentro de train; genera `folds_v1.csv` y `reports/folds_report.md`. Expone `load_train_test()` y `predefined_cv()` para que todos los modelos usen la misma partición.
  * `features.py`: Representación de texto (TF-IDF de título, descripción y tags) y de metadatos como `ColumnTransformer`; `build_preprocessor("text" | "metadata" | "combined", svd_components=None)` para el ablation study (`reports/text_features.md`).
  * `leakage_check.py`: Compara validación aleatoria vs. agrupada por canal con un modelo TF-IDF + regresión logística (`reports/leakage_check.md`).
  * `baselines.py`: Modelos baseline (Dummy, LogReg + TF-IDF, Naive Bayes); genera `reports/baselines.md`.
  * `models.py`: Modelos tabulares, combinado y ablation study (Random Forest, LightGBM, LogReg combinado); genera `reports/models.md`.
  * `tuning.py`: Ajuste de hiperparámetros con `RandomizedSearchCV` y evaluación final en test; genera `reports/tuning.md`, `reports/final_model.md`, `reports/test_predictions.csv` y `models/final.joblib`.
  * `paired_comparison.py`: Comparación pareada fold a fold de la regresión logística y LightGBM ajustados (diferencias por fold y prueba t pareada); genera `reports/paired_comparison.md`.
  * `evaluation.py`: Curvas ROC/PR, matriz de confusión y tabla de métricas completa; genera `reports/evaluation.md` y las figuras en `reports/figures/`. Si `models/final.joblib` no existe lo re-entrena automáticamente.
  * `interpretability.py`: SHAP values con `LinearExplainer`, importancia de features por clase e análisis cualitativo de falsos negativos/positivos; genera `reports/interpretability.md` y figuras SHAP en `reports/figures/`.
  * `additional_inference.py`: Aplica el modelo final a los conjuntos adicionales (sin videos del ground truth) y lo compara con el clasificador original (PI2 y PI3); genera `reports/additional_consistency.md` y `outputs/additional_predictions.csv.gz` (este último excluido de git).
* `latex/`: Informe en formato IEEE (`InformeP1.tex`, `references.bib`, `images/`).
  * `reports/`: Reportes Markdown generados automáticamente por los scripts anteriores.
  * `reports/figures/`: Figuras PNG (ROC, PR, matriz de confusión, SHAP) listas para el paper IEEE.
  * `models/`: Modelo final serializado (`final.joblib`, ~2.5 MB). Excluido de git por `.gitignore`; regenerar con `python -m modeling.tuning` o `python -m modeling.evaluation`.
* `dataset_disturbed_youtube _for_kids/`: Directorio con datos tabulares y scripts de conversión.
  * `convert_json_to_csv.py`: Script de utilidad para transformar los archivos JSON crudos a formato CSV.
  * `dataset_groundtruth.csv`: Dataset principal anotado manualmente por humanos (4,797 videos, 4 clases).
  * `dataset_consolidado_parts/`: Corpus particionado en 10 archivos comprimidos (`part_01.csv.gz` a `part_10.csv.gz`) con 844,702 videos únicos.
  * `elsagate_related_videos_parts/`, `other_child_related_videos_parts/`, `random_videos_parts/`: Particiones de datos adicionales comprimidos en `.json.gz`.
* `dataset_prueba/`: Módulos experimentales y utilidades complementarias para extracción de señales audiovisuales.
* `requirements.txt`: Especificación de librerías y dependencias de Python para reproducibilidad.
* `.gitignore`: Reglas de exclusión para datos pesados (>50MB, CSV completo de 1GB, JSONs crudos de 3.4GB, binarios y temporales).
* `README.md`: Descripción general, objetivos, integrantes, estructura y guía de uso del proyecto.

---

## Obtención y Preparación del Dataset

El dataset *Disturbed YouTube for Kids* proviene de la investigación de Papadamou et al. (2020) y se encuentra alojado con acceso restringido en Zenodo:
* **DOI:** [https://doi.org/10.5281/zenodo.3632781](https://doi.org/10.5281/zenodo.3632781)
* **Licencia y Restricciones:** El acceso al dataset en Zenodo es restringido y su uso está sujeto a las condiciones de los autores y a los Términos de Servicio de la YouTube Data API.
* **Estado actual del repositorio:** El repositorio es público e **incluye** los archivos del dataset dentro de `dataset_disturbed_youtube _for_kids/` (`dataset_groundtruth.csv`, el consolidado particionado en `dataset_consolidado_parts/` y los JSON crudos comprimidos en `.json.gz`, ~1.1 GB), para que el curso pueda reproducir el análisis sin pasos adicionales. Se incluyen únicamente con fines académicos; quien desee reutilizar los datos fuera de este proyecto debe solicitar acceso en Zenodo y citar a Papadamou et al. (2020).

### Dónde están los datos
No es necesario descargar nada: al clonar el repositorio, los scripts y notebooks encuentran los datos en sus rutas por defecto.
1. Ground truth (entrenamiento y evaluación):
   ```text
   dataset_disturbed_youtube _for_kids/dataset_groundtruth.csv
   ```
   *(Los notebooks de `exploratory_data_analysis/` lo leen con la ruta relativa `../dataset_disturbed_youtube _for_kids/dataset_groundtruth.csv`).*
2. Conjuntos adicionales (consistencia, PI2 y PI3): `dataset_disturbed_youtube _for_kids/dataset_consolidado_parts/part_01.csv.gz` a `part_10.csv.gz`.
3. Si en el futuro se retiran los datos del repositorio, basta con descargarlos desde Zenodo y colocarlos en esas mismas rutas.
4. No se deben commitear archivos generados pesados: el CSV consolidado descomprimido de 1 GB, los `.json` sin comprimir, ni `modeling/outputs/` o `modeling/models/` (ya excluidos en `.gitignore`).

---

## Estructura de Datasets

### 1. Dataset Ground Truth (Anotación Humana)
* **Archivo:** `dataset_groundtruth.csv`
* **Registros:** 4,797 videos etiquetados manualmente.
* **Clases:** `suitable` (apto), `disturbing` (perturbador), `restricted` (restringido), `irrelevant` (no relevante).
* **Uso:** Entrenamiento, validación y evaluación de modelos supervisados.

### 2. Dataset Consolidado Completo (Particionado y Comprimido)
* **Carpeta:** `dataset_consolidado_parts/`
* **Archivos:** `part_01.csv.gz` hasta `part_10.csv.gz`
* **Registros:** 844,702 videos únicos (desduplicados).
* **Uso:** Inferencia masiva, análisis exploratorio global y modelos de lenguaje.

### 3. Datos Crudos JSON (Comprimidos)
* **Ubicación:** `dataset_disturbed_youtube _for_kids/` (`groundtruth_videos.json.gz`, `popular_videos.json.gz` y las carpetas `elsagate_related_videos_parts/`, `other_child_related_videos_parts/` y `random_videos_parts/`).
* Contiene los archivos originales recolectados en formato JSON comprimidos en `.json.gz` (groundtruth, popular, elsagate_related, other_child_related y random).

---

## Estado del Proyecto

| Fase | Responsable | Estado | Entregable |
|---|---|---|---|
| Preprocesamiento de datos | Mariela | ✅ Completo | `data_processing/preprocessing.py`, `dataset_features_v1.csv` |
| Feature engineering de texto | Mariela | ✅ Completo | `modeling/features.py` |
| Split y validación | Adrián | ✅ Completo | `modeling/splits.py`, `folds_v1.csv`, `reports/folds_report.md` |
| Modelos baseline | Leo | ✅ Completo | `modeling/baselines.py`, `reports/baselines.md` |
| Modelos tabulares y combinado | Leo | ✅ Completo | `modeling/models.py`, `modeling/tuning.py`, `reports/models.md`, `reports/tuning.md` |
| Evaluación y métricas | Mafer | ✅ Completo | `modeling/evaluation.py`, `reports/evaluation.md`, figuras ROC/PR/CM |
| Interpretabilidad y errores | Mafer | ✅ Completo | `modeling/interpretability.py`, `reports/interpretability.md`, figuras SHAP |
| Pruebas en datasets adicionales | Adrián | ✅ Completo | `modeling/additional_inference.py`, `reports/additional_consistency.md` |
| Discusión, ética y limitaciones | Adrián | ✅ Completo | Secciones VII y VIII del paper (`latex/InformeP1.tex`) |
| Redacción e integración del paper | Mariela (coord.) | ✅ Completo | `latex/InformeP1.tex`: 10 secciones, 12 páginas IEEE (pendiente revisión final del equipo) |
| Repositorio y reproducibilidad | Leo | ✅ Completo | README, `requirements.txt`, secuencia de reproducción |
| Presentación P2 | Todo el equipo | ✅ Completo | Slides |

---

## Resultados del Modelo Final (Etapa 2)

**Modelo:** Regresión Logística combinada (TF-IDF + metadatos), `C=1.16`, `class_weight="balanced"`.
Seleccionado por F1 macro en validación cruzada agrupada por canal (5 folds). Evaluado **una sola vez** sobre 955 videos de test (canales no vistos en entrenamiento).

### Comparación de modelos (CV en train, 3 842 videos)

| Modelo | Vista | F1 macro (CV) | Recall disturbing | Recall restricted |
|---|---|---:|---:|---:|
| Dummy (clase mayoritaria) | texto | 0.145 ± 0.004 | 0.000 | 0.000 |
| Naive Bayes + TF-IDF | texto | 0.659 ± 0.033 | 0.670 | 0.254 |
| LogReg + TF-IDF | texto | 0.672 ± 0.025 | 0.637 | 0.359 |
| Solo texto (LightGBM + SVD 200) | texto | 0.625 ± 0.002 | 0.609 | 0.157 |
| Random Forest (metadatos) | metadatos | 0.575 ± 0.027 | 0.457 | 0.323 |
| LightGBM (metadatos) | metadatos | 0.580 ± 0.020 | 0.496 | 0.326 |
| LightGBM combinado (SVD 200 + metadatos) | combinado | 0.682 ± 0.020 | 0.621 | 0.331 |
| **LogReg combinado (TF-IDF + metadatos)** | combinado | **0.710 ± 0.023** | **0.664** | **0.566** |

### Métricas en test (955 videos, canales no vistos)

| Métrica | Valor |
|---|---:|
| F1 macro | **0.725** |
| Recall macro | 0.724 |
| Accuracy | 0.775 |
| AUC macro ROC (OvR) | **0.914** |
| AP macro Precision-Recall | **0.772** |

| Clase | Precision | Recall | F1 | AUC | AP | Videos |
|---|---:|---:|---:|---:|---:|---:|
| disturbing | 0.801 | 0.714 | 0.755 | 0.929 | 0.848 | 192 |
| irrelevant | 0.794 | 0.837 | 0.815 | 0.921 | 0.873 | 363 |
| restricted | 0.500 | 0.550 | 0.524 | 0.878 | 0.483 | 80 |
| suitable | 0.815 | 0.797 | 0.806 | 0.929 | 0.885 | 320 |

Matriz de confusión (filas = real, columnas = predicción):

| Real \ Predicho | disturbing | irrelevant | restricted | suitable |
|---|---:|---:|---:|---:|
| disturbing | 137 | 17 | 9 | 29 |
| irrelevant | 16 | 304 | 24 | 19 |
| restricted | 4 | 22 | 44 | 10 |
| suitable | 14 | 40 | 11 | 255 |

> Los reportes completos y las figuras (curvas ROC/PR, matrices de confusión, SHAP) se encuentran en `modeling/reports/` y `modeling/reports/figures/`.

### Consistencia en conjuntos adicionales (PI2 y PI3)

El modelo final se aplicó a 839,905 videos que **no** están en el ground truth y se comparó con la predicción binaria del clasificador original (*inapropiado* = `disturbing` o `restricted`). No hay etiquetas humanas: mide consistencia, no eficacia.

| Conjunto | Videos | % inapropiado (modelo) | % inapropiado (original) | Kappa |
|---|---:|---:|---:|---:|
| elsagate_related | 230,873 | 21.78 | 0.55 | −0.002 |
| other_child_related | 137,043 | 11.92 | 0.43 | −0.007 |
| random | 462,382 | 6.74 | 0.73 | 0.000 |
| popular | 9,607 | 12.00 | 0.00 | 0.000 |

El modelo conserva el orden de riesgo esperado (Elsagate > aleatorios), pero genera muchas más alertas que el clasificador original, sobre todo en `restricted` sobre contenido popular no infantil (H2 y H3 no se cumplen). Detalle en `modeling/reports/additional_consistency.md` y en la Sección VI.D del informe.

---

## Informe (paper IEEE)

* `latex/InformeP1.tex`: informe completo de la Etapa 2 (12 páginas, plantilla IEEE de conferencias), con `latex/references.bib` y las figuras en `latex/images/`.
* Compilar desde `latex/` (o subir la carpeta a Overleaf):
  ```bash
  cd latex
  pdflatex InformeP1.tex && bibtex InformeP1 && pdflatex InformeP1.tex && pdflatex InformeP1.tex
  ```
* Las figuras y tablas del EDA que no entraron en el límite de 12 páginas siguen en `exploratory_data_analysis/images/` y `exploratory_data_analysis/EDA.md`.

---

## Análisis Exploratorio de Datos (EDA)

* `exploratory_data_analysis/eda_analysis.ipynb`: Notebook principal de análisis exploratorio (figuras originales con alta resolución, imágenes en `images/`, usadas en `EDA.md` y la presentación).
* `exploratory_data_analysis/eda_analysis_overleaf.ipynb`: Notebook especializado para el paper (genera las figuras a tamaño final IEEE dos columnas en `images_overleaf/`, que se suben al proyecto de Overleaf).
* `exploratory_data_analysis/EDA.md`: Documento consolidado del informe de análisis exploratorio.

---

## Instrucciones para Ejecutar el Código

### 1. Requisitos de Entorno
* **Python:** 3.10 o superior (el código usa sintaxis de tipos `X | None`, no compatible con versiones anteriores).
* **scikit-learn:** 1.7.2 exacto (`requirements.txt`), porque la partición de `StratifiedGroupKFold` cambia entre versiones. Probado con pandas 2.2 y 3.0.
* Se recomienda utilizar un entorno virtual (`venv` o `conda`).

### 2. Creación del Entorno Virtual e Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/ADRIAN0819/Machine-Learning-Proyect.git
cd Machine-Learning-Proyect

# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# En Linux / macOS:
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Orden de Ejecución de los Notebooks

1. **Notebook Principal (`exploratory_data_analysis/eda_analysis.ipynb`):**
   * Es el notebook que debe ejecutarse en primer lugar para reproducir el análisis exploratorio de datos completo.
   * Carga `dataset_groundtruth.csv`, ejecuta la caracterización estadística, genera las tablas numéricas y guarda las 14 figuras originales en la carpeta `images/`.
   * Comando de ejecución:
     ```bash
     jupyter notebook exploratory_data_analysis/eda_analysis.ipynb
     ```
2. **Notebook para el Paper (`exploratory_data_analysis/eda_analysis_overleaf.ipynb`):**
   * Solo es necesario ejecutarlo si se desea regenerar las figuras con escala tipográfica 1:1 adaptadas a columnas de LaTeX/Overleaf (formato IEEE Transactions).
   * Guarda sus resultados en la carpeta `images_overleaf/`.
3. **Preprocesamiento (`data_processing/preprocesamiento_features_v1.ipynb`):**
   * Genera `data_processing/dataset_features_v1.csv` desde `dataset_groundtruth.csv`.
   * Imputa con estadísticas ajustadas solo en train y crea un fold estratificado y agrupado por canal (3,842 train / 955 test con scikit-learn 1.7.2). `video_id`, `channel_id` y `split` son metadatos: `channel_id` sirve para agrupar la validación cruzada dentro de train y ninguna de las tres debe usarse como feature.
4. **Partición y validación (`modeling/`):**
   * `python -m modeling.splits` — regenera `modeling/folds_v1.csv` y el reporte de folds.
   * `python -m modeling.features` — genera el reporte de la representación de texto.
   * `python -m modeling.leakage_check` — reproduce la comparación validación aleatoria vs. agrupada por canal.
   * `python -m modeling.baselines` — entrena y evalúa los modelos baseline (CV sobre train).
   * `python -m modeling.models` — entrena los modelos tabulares, combinado y ablation study (CV sobre train).
   * `python -m modeling.tuning` — ajuste de hiperparámetros y evaluación final en test; guarda `modeling/models/final.joblib`.
   * `python -m modeling.paired_comparison` — compara fold a fold los dos modelos ajustados (~1 min).
   * `python -m modeling.evaluation` — curvas ROC/PR, matriz de confusión y tabla de métricas completa; genera figuras en `modeling/reports/figures/`. Si `final.joblib` no existe, lo re-entrena automáticamente.
   * `python -m modeling.interpretability` — SHAP values, importancia de features y análisis cualitativo de errores; genera figuras SHAP en `modeling/reports/figures/`.
   * `python -m modeling.additional_inference --model modeling/models/final.joblib` — evalúa la consistencia en los conjuntos adicionales (~4 min para 839,905 videos; usar `--demo --limit-per-part 5000` para una prueba rápida).
   * Ejecutar todos los comandos desde la raíz del repositorio.

### Reproducir todos los resultados de la Etapa 2

Con el entorno activado y desde la raíz del repositorio:

```bash
python -m modeling.splits            # particion agrupada por canal (folds_v1.csv)
python -m modeling.leakage_check     # validacion aleatoria vs. agrupada
python -m modeling.baselines         # Dummy, LogReg y Naive Bayes
python -m modeling.models            # RF, LightGBM, combinados y ablation
python -m modeling.tuning            # ajuste de hiperparametros + models/final.joblib
python -m modeling.paired_comparison # RL vs. LightGBM fold a fold
python -m modeling.evaluation        # metricas de test, ROC/PR y matriz de confusion
python -m modeling.interpretability  # SHAP y analisis de errores
python -m modeling.additional_inference --model modeling/models/final.joblib  # PI2 y PI3
```

Cada script escribe su reporte en `modeling/reports/`. Con scikit-learn 1.7.2 se reproducen las métricas del informe (F1 macro en test 0.725); en otro hardware pueden variar en el tercer decimal métricas como el AP.

> **Windows:** usar el Python oficial de python.org (por ejemplo `py -3.13 -m venv venv`), no el de MSYS2/Git Bash, que no tiene ruedas precompiladas de scikit-learn.

### 4. Uso de la partición en los modelos

```python
from modeling.splits import load_train_test, predefined_cv
from modeling.features import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

X_train, y_train, X_test, y_test, meta_train, meta_test = load_train_test()
pipeline = make_pipeline(build_preprocessor("combined"), LogisticRegression(max_iter=3000, class_weight="balanced"))
param_grid = {"logisticregression__C": [0.1, 1, 10]}
search = GridSearchCV(pipeline, param_grid, cv=predefined_cv(meta_train), scoring="f1_macro")
search.fit(X_train, y_train)   # el conjunto de prueba solo se usa al final
```

---

## Guía de Uso en Python

### Cargar el Dataset Ground Truth

```python
import pandas as pd

# Cargar dataset de entrenamiento curado
df_gt = pd.read_csv('dataset_disturbed_youtube _for_kids/dataset_groundtruth.csv')
print(f'Total registros: {len(df_gt)}')
print(df_gt['classification_label'].value_counts())
```

### Cargar el Dataset Consolidado Completo (.csv.gz)

Pandas puede leer directamente archivos comprimidos con gzip sin necesidad de descomprimirlos en disco:

```python
import glob
import pandas as pd

# Obtener rutas de todas las partes comprimidas
partes = sorted(glob.glob('dataset_disturbed_youtube _for_kids/dataset_consolidado_parts/*.csv.gz'))

# Unir todas las partes en un unico DataFrame
df_completo = pd.concat(
    [pd.read_csv(f, compression='gzip') for f in partes],
    ignore_index=True
)

print(f'Total de videos cargados: {len(df_completo):,}')
```

### Leer Archivos JSON Comprimidos (.json.gz)

Para leer línea por línea los archivos JSON comprimidos:

```python
import gzip
import json

ruta_gz = 'dataset_disturbed_youtube _for_kids/groundtruth_videos.json.gz'

with gzip.open(ruta_gz, 'rt', encoding='utf-8') as f:
    for linea in f:
        video = json.loads(linea)
        titulo = video.get('snippet', {}).get('title')
        print(titulo)
        break
```
