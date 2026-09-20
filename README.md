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
* **Licencia y Restricciones:** Debido a los Términos de Servicio de la YouTube Data API y a las condiciones de distribución académica de Zenodo, los datos crudos y consolidados no deben ser redistribuidos públicamente en repositorios abiertos. Por ello, este repositorio **NO** incluye los archivos de datos masivos.

### Dónde colocar los datos
Para ejecutar los notebooks de análisis:
1. Descargar el archivo `dataset_groundtruth.csv` desde Zenodo (o solicitar acceso a los autores).
2. Colocar `dataset_groundtruth.csv` dentro de la carpeta:
   ```text
   dataset_disturbed_youtube _for_kids/dataset_groundtruth.csv
   ```
   *(Esta es la ruta relativa `../dataset_disturbed_youtube _for_kids/dataset_groundtruth.csv` que esperan por defecto los notebooks dentro de `exploratory_data_analysis/`).*
3. Los archivos `.json` crudos o el CSV consolidado de 1 GB no deben commitearse; `.gitignore` ya está configurado para excluirlos automáticamente.

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
* **Carpeta:** `DataSet_Disturbed_YouTube_Kids/`
* Contiene los archivos originales recolectados en formato JSON comprimidos en `.json.gz` (groundtruth, popular, elsagate_related, other_child_related y random).

---

## Análisis Exploratorio de Datos (EDA)

* `exploratory_data_analysis/eda_analysis.ipynb`: Notebook principal de análisis exploratorio (figuras originales con alta resolución, imágenes en `images/`, usadas en `EDA.md` y la presentación).
* `exploratory_data_analysis/eda_analysis_overleaf.ipynb`: Notebook especializado para el paper (genera las figuras a tamaño final IEEE dos columnas en `images_overleaf/`, que se suben al proyecto de Overleaf).
* `exploratory_data_analysis/EDA.md`: Documento consolidado del informe de análisis exploratorio.

---

## Instrucciones para Ejecutar el Código

### 1. Requisitos de Entorno
* **Python:** 3.10 o superior (compatible con Python 3.8+).
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
