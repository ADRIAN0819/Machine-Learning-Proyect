# Proyecto Machine Learning: Deteccion de Videos No Aptos para Ninos en YouTube

Repositorio para la clasificacion y deteccion de videos inapropiados o perturbadores dirigidos a ninos en YouTube (fenomeno Elsagate), basado en metadatos, texto (NLP) y metricas de interaccion.

---

## Estructura de Datasets

### 1. Dataset Ground Truth (Anotacion Humana)
* **Archivo:** `dataset_groundtruth.csv`
* **Registros:** 4,797 videos etiquetados manualmente.
* **Clases:** `suitable` (apto), `disturbing` (perturbador), `restricted` (restringido), `irrelevant` (no relevante).
* **Uso:** Entrenamiento, validacion y evaluacion de modelos supervisados.

### 2. Dataset Consolidado Completo (Particionado y Comprimido)
* **Carpeta:** `dataset_consolidado_parts/`
* **Archivos:** `part_01.csv.gz` hasta `part_10.csv.gz`
* **Registros:** 844,702 videos unicos (desduplicados).
* **Uso:** Inferencia masiva, analisis exploratorio global y modelos de lenguaje.

### 3. Datos Crudos JSON (Comprimidos)
* **Carpeta:** `DataSet_Disturbed_YouTube_Kids/`
* Contiene los archivos originales recolectados en formato JSON comprimidos en `.json.gz` (groundtruth, popular, elsagate_related, other_child_related y random).

---

## Guia de Uso en Python

### Cargar el Dataset Ground Truth

```python
import pandas as pd

# Cargar dataset de entrenamiento curado
df_gt = pd.read_csv('dataset_groundtruth.csv')
print(f'Total registros: {len(df_gt)}')
print(df_gt['classification_label'].value_counts())
```

### Cargar el Dataset Consolidado Completo (.csv.gz)

Pandas puede leer directamente archivos comprimidos con gzip sin necesidad de descomprimirlos en disco:

```python
import glob
import pandas as pd

# Obtener rutas de todas las partes comprimidas
partes = sorted(glob.glob('dataset_consolidado_parts/*.csv.gz'))

# Unir todas las partes en un unico DataFrame
df_completo = pd.concat(
    [pd.read_csv(f, compression='gzip') for f in partes],
    ignore_index=True
)

print(f'Total de videos cargados: {len(df_completo):,}')
```

### Leer Archivos JSON Comprimidos (.json.gz)

Para leer linea por linea los archivos JSON comprimidos:

```python
import gzip
import json

ruta_gz = 'DataSet_Disturbed_YouTube_Kids/groundtruth_videos.json.gz'

with gzip.open(ruta_gz, 'rt', encoding='utf-8') as f:
    for linea in f:
        video = json.loads(linea)
        titulo = video.get('snippet', {}).get('title')
        print(titulo)
        break
```

---

## Requisitos

* Python 3.8+
* pandas
* scikit-learn
