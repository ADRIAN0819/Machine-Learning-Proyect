# Informe de Análisis Exploratorio de Datos (EDA)
## Dataset: Disturbed YouTube for Kids - Clasificación de Videos Inapropiados para Niños

**Autor:** Equipo de Proyecto (Machine Learning)  
**Fecha:** 13 de septiembre de 2026  
**Proyecto:** Machine Learning Project - Clasificación de contenido inapropiado para niños

---

## 1. Introducción y Formulación del Problema

Este informe presenta el análisis exploratorio de datos (EDA) realizado sobre el dataset "Disturbed YouTube for Kids", el cual contiene videos de YouTube clasificados según su apropiación para niños pequeños. El objetivo principal de este análisis es comprender la estructura, distribución y características de los datos para el posterior entrenamiento de un modelo de machine learning capaz de detectar automáticamente contenido inapropiado.

### 1.1 Descripción y motivación del problema

El acceso de niños pequeños a plataformas de video como YouTube es masivo y creciente. Actualmente se suben más de 500 horas de video por minuto a YouTube, lo que ilustra la escala del contenido disponible (Brandwatch, 2026). Aunque existe una gran cantidad de videos dirigidos al público infantil, muchos materiales con apariencia infantil contienen escenas o temas inadecuados: violencia, lenguaje ofensivo, sexualización o imágenes perturbadoras. Este problema fue identificado como **Elsagate**, donde videos con personajes de dibujos animados populares (Peppa Pig, Spider-Man, Elsa, etc.) mostraban situaciones grotescas u obscenas dirigidas a niños de 2 a 6 años (United States Cybersecurity Magazine, s. f.). Estas producciones emplean entornos coloridos y palabras clave infantiles, pero en realidad incluyen contenidos potencialmente dañinos o perturbadores para los niños, como mutilaciones, inyecciones o escenas de miedo (United States Cybersecurity Magazine, s. f.).

Debido al enorme flujo de contenido, con miles de millones de videos subidos y cientos de miles nuevos cada día, la revisión humana exhaustiva es inviable. Plataformas como YouTube han empezado a usar aprendizaje automático para ayudar en la moderación de contenidos y aplicar restricciones de edad automáticamente (YouTube, 2020). En este contexto, se propone desarrollar un modelo de aprendizaje supervisado capaz de detectar y clasificar automáticamente videos potencialmente inapropiados para niños en YouTube, usando sus metadatos y textos asociados.

El proyecto dispone de un conjunto de datos de referencia con 4,797 videos etiquetados manualmente en cuatro clases: `suitable`, `disturbing`, `restricted` e `irrelevant` (Papadamou et al., 2020a), cuyas definiciones se detallan en la Sección 2.3. Además, se cuenta con un corpus más amplio para extender el análisis a gran escala. El análisis exploratorio de los datos revela patrones en títulos, etiquetas, duración, categorías de YouTube y métricas de interacción que diferencian parcialmente estas clases. El principal desafío del proyecto es distinguir contenido infantil legítimo de aquellos videos engañosos que, bajo apariencia de material infantil (personajes conocidos, colores, juguetes), ocultan elementos inadecuados (United States Cybersecurity Magazine, s. f.). Esta tarea es compleja porque los creadores malintencionados pueden modificar ligeramente el contenido y las plataformas públicas evolucionan continuamente.

La motivación del proyecto combina varios factores clave:

- Alto volumen de contenido infantil en YouTube, lo que hace prácticamente imposible su supervisión manual total.
- Dificultad para filtrar videos engañosos que parecen inocuos pero contienen material perturbador.
- Necesidad de herramientas automáticas que asistan la identificación y clasificación de videos para niños.
- Importancia social de proteger a los niños de contenidos inapropiados y facilitar el trabajo de padres, tutores y plataformas.

El objetivo es crear una herramienta de apoyo que agilice la detección de videos sospechosos, analice volúmenes masivos de información y contribuya a una clasificación más eficiente. No se busca reemplazar la supervisión humana, sino ayudarla haciendo más rápida y precisa la preclasificación de contenido potencialmente problemático.

### 1.2 Objetivos del proyecto

**Objetivo general.** Desarrollar un modelo de aprendizaje supervisado que clasifique automáticamente videos de YouTube según su idoneidad para niños pequeños (`suitable`, `irrelevant`, `restricted`, `disturbing`) a partir de sus metadatos, como herramienta de apoyo a la moderación de contenido.

**Objetivos específicos.**
1. Caracterizar el dataset *Disturbed YouTube for Kids* mediante un análisis exploratorio: estructura, valores faltantes, outliers, desbalance de clases, relaciones entre variables y riesgos de fuga de datos.
2. Diseñar el preprocesamiento y la ingeniería de características a partir del texto y de las métricas de interacción.
3. Entrenar y comparar un baseline y al menos dos enfoques de modelado, con una validación que agrupe por canal.
4. Evaluar el desempeño con métricas apropiadas para clases desbalanceadas, con énfasis en el recall de `disturbing` y `restricted`, y analizar los errores.
5. Comprobar la consistencia de las predicciones del modelo sobre los conjuntos adicionales, que no tienen etiquetas humanas fuera del ground truth.

### 1.3 Definición de la tarea de Machine Learning

Se plantea un modelo de aprendizaje supervisado que, a partir de los metadatos y textos asociados a cada video, prediga su clase de adecuación para niños. Para cada video $v$ se define un vector de características $\mathbf{x}_v$ construido a partir de sus metadatos (título, descripción, etiquetas, categoría, duración, calidad, visualizaciones, likes, dislikes y comentarios), que representan distintos aspectos del video:

- **Texto asociado:** título, descripción y etiquetas (tags).
- **Categoría:** categoría de YouTube (por ejemplo, Educación o Entretenimiento).
- **Características del video:** duración y calidad de la imagen (definición).
- **Métricas de interacción:** visualizaciones, likes, dislikes y comentarios, que reflejan la popularidad y la respuesta del público.

El identificador y el nombre del canal **no** se incluyen como características, para evitar que el modelo memorice rasgos propios de cada canal (ver Sección 6.4); el canal se usa únicamente para agrupar la partición de los datos.

A partir de estas características, el modelo aprenderá una función de predicción. Formalmente, la tarea se expresa como:

$$\hat{y}_v = f(\mathbf{x}_v)$$

donde $\hat{y}_v \in \mathcal{Y}$ es la clase predicha para el video $v$, con $\mathcal{Y} = \{\text{suitable}, \text{disturbing}, \text{restricted}, \text{irrelevant}\}$, y $f\colon \mathbb{R}^n \to \mathcal{Y}$ opera sobre las características ya vectorizadas. El modelo buscará que $\hat{y}_v$ coincida con la etiqueta real $y_v$ del conjunto de entrenamiento. Inicialmente se aborda como un problema de **clasificación multiclase supervisada**, dado que se cuenta con etiquetas manuales.

Para entrenar el modelo se emplean los 4,797 videos etiquetados. La función $f$ puede implementarse con distintos algoritmos de clasificación (por ejemplo, regresión logística multiclase, árboles de decisión, ensamblajes o redes neuronales). El modelo entrenado se evaluará en un conjunto de prueba independiente, reservado del ground truth. Se prestará especial atención al desempeño en las clases `disturbing` y `restricted`, ya que clasificar erróneamente estos videos como seguros tendría consecuencias más graves.

Como alternativa a la clasificación multiclase, también se explorará la idea de predecir un puntaje continuo de nivel de inapropiación (por ejemplo, entre 0 y 1). En este caso, la función $f$ sería de regresión y podría formularse como $y_v \in [0,1]$ en lugar de asignar categorías discretas. Sin embargo, inicialmente se implementará la tarea como clasificación para ajustarse a las etiquetas disponibles.

Los pasos generales del pipeline serán:

1. Extracción automática de características $\mathbf{x}_v$ de cada video (a partir de metadatos y texto).
2. Preprocesamiento y normalización de datos.
3. Entrenamiento del modelo de aprendizaje supervisado con las etiquetas conocidas.
4. Evaluación con validación cruzada y en el conjunto de prueba separado.
5. Análisis de resultados (matriz de confusión, métricas) y ajuste de parámetros según corresponda.

Adicionalmente, aunque la tarea principal se centra en metadatos, se podría considerar como extensión la incorporación de características audiovisuales extraídas directamente del contenido (por ejemplo, tasa de cambios de escena, movimiento, estadísticas de audio). Esto podría enriquecer el análisis, pero representa un esfuerzo adicional significativo, por lo que por ahora se priorizan las fuentes de información más accesibles: texto, etiquetas y métricas del video.

Las métricas elegidas para evaluar el desempeño son:

- **Exactitud (accuracy):** proporción de videos correctamente clasificados.
- **Precisión y exhaustividad (recall) por clase:** en particular en `disturbing` y `restricted`, que miden la capacidad de identificar correctamente videos problemáticos.
- **Puntuación F1:** media armónica entre precisión y recall, para balancear ambos indicadores.
- **Matriz de confusión:** para visualizar patrones de error entre clases.

### 1.4 Preguntas de investigación e hipótesis

Se plantean las siguientes preguntas de investigación, cada una con su hipótesis:

**Pregunta 1 (central):** ¿Es posible predecir si un video dirigido al público infantil es apropiado, perturbador o restringido utilizando únicamente metadatos (título, descripción, tags), duración y métricas de interacción (vistas, likes, comentarios), sin necesidad de procesar el video o el audio directamente?

- **Hipótesis 1:** las señales textuales y de interacción contienen información suficiente para distinguir entre clases con un desempeño claramente superior al de un modelo base ingenuo (por ejemplo, uno que siempre prediga la clase mayoritaria, `irrelevant`).

**Pregunta 2 (generalización):** excluyendo los videos que ya forman parte del ground truth de entrenamiento (para evitar fuga de datos), ¿qué proporción de los videos nuevos de `elsagate_related`, un escenario de alto riesgo, predice el modelo como inapropiados, y qué tan consistente es con la predicción del clasificador original de los autores?

- **Hipótesis 2:** como fuera del ground truth el clasificador original predijo como inapropiado solo 1 de 220 videos de las primeras 1,000 líneas de este conjunto (ver Sección 5.1), se espera que el modelo también prediga una proporción baja, y las discrepancias se revisarán de forma cualitativa.

**Pregunta 3 (comprobación de consistencia):** excluidos los solapamientos con el ground truth, ¿mantiene el modelo una proporción baja de videos predichos como inapropiados en `other_child_related`, `random_videos` y `popular_videos`, donde el clasificador original no predijo ninguno fuera del ground truth?

- **Hipótesis 3:** se espera una proporción baja y coherente con la del clasificador original, aunque podría aumentar levemente por el cambio de distribución entre conjuntos.

Como fuera del ground truth no existen etiquetas humanas, las preguntas 2 y 3 se responden como comprobaciones de consistencia y no como medidas de recall o de tasa de falsos positivos (ver Secciones 5 y 8).

---

## 2. Descripción del Dataset

### 2.1 Fuente y Estructura General

El dataset utilizado proviene de Papadamou et al. (2020a), "Disturbed YouTube for Kids: Characterizing and Detecting Inappropriate Videos Targeting Young Children", presentado en la 14th International AAAI Conference on Web and Social Media (ICWSM 2020) y publicado en Zenodo (Papadamou et al., 2020b). Los datos fueron recolectados mediante la YouTube Data API (Google, s. f.) y consisten en metadatos de video, no en el contenido audiovisual en sí.

El equipo trabaja con tres formas del mismo dataset:

- `dataset_groundtruth.csv`: versión tabular del ground truth, fuente principal de entrenamiento y de evaluación formal.
- `dataset_consolidado_parts/`: 844,702 videos únicos, particionados en 10 archivos `.csv.gz`, que integran los datasets adicionales.
- Archivos JSON crudos comprimidos: formato original de la API, previo a la conversión a CSV. El análisis breve de los datasets adicionales de este informe se realizó sobre estos archivos.

**Dataset Anotado (Ground Truth):**
- **Archivo:** `dataset_groundtruth.csv`
- **Total de videos:** 4,797 videos, sin duplicados (verificado por `video_id` único), y 22 variables
- **Etiquetado:** 100% manual por humanos
- **Propósito:** Entrenamiento y evaluación formal (split train/test y validación cruzada)

**Datasets Adicionales (Comprobación de Consistencia):**
- **elsagate_related_videos:** ~233K videos (5 partes comprimidas)
- **other_child_related_videos:** ~155K videos (3 partes comprimidas)
- **random_videos:** ~482K videos (8 partes comprimidas)
- **popular_videos:** ~11K videos (archivo único comprimido)
- **Propósito:** Comprobación de consistencia y predicciones con el modelo entrenado (sin etiquetas humanas, salvo los videos que ya están en el ground truth)

### 2.2 Variables y Tipos de Datos

El `dataset_groundtruth.csv` contiene 22 variables. La siguiente tabla indica el tipo de dato y el número de valores faltantes de cada una:

**Tabla 0: Variables del dataset de ground truth (4,797 videos)**

| Variable | Tipo | Faltantes | Descripción |
|----------|------|----------:|-------------|
| `video_id` | texto (identificador) | 0 | ID único del video |
| `title` | texto | 0 | Título del video |
| `description` | texto | 506 | Descripción del video |
| `channel_id` | texto (identificador) | 0 | ID único del canal |
| `channel_title` | texto | 0 | Nombre del canal |
| `published_at` | fecha y hora (ISO 8601) | 0 | Fecha de publicación |
| `category_id` | categórica (17 valores) | 0 | Categoría de YouTube |
| `tags` | texto (lista) | 883 | Etiquetas asociadas al video |
| `duration_iso` | texto (ISO 8601, ej. PT2M30S) | 1 | Duración en el formato original |
| `duration_seconds` | numérica (float) | 1 | Duración en segundos |
| `definition` | categórica (hd/sd) | 1 | Calidad del video (78.1% hd, 21.8% sd) |
| `caption` | booleana | 1 | Si el video tiene subtítulos |
| `licensed_content` | booleana | 1 | Si el contenido tiene licencia comercial |
| `view_count` | numérica (float) | 3 | Número de visualizaciones |
| `like_count` | numérica (float) | 128 | Número de likes |
| `dislike_count` | numérica (float) | 128 | Número de dislikes |
| `comment_count` | numérica (float) | 228 | Número de comentarios |
| `favorite_count` | numérica (float) | 0 | Número de favoritos (YouTube ya no lo usa desde 2015) |
| `classification_label` | categórica (4 clases) | 0 | **Variable objetivo** |
| `prediction` | categórica | 4,797 (100%) | Predicción del clasificador original de los autores |
| `is_ground_truth` | booleana (constante = 1) | 0 | Confirma que el registro proviene de anotación manual |
| `source_dataset` | categórica (constante = "groundtruth") | 0 | Origen del registro |

### 2.3 Variable Objetivo

La variable objetivo es `classification_label`, que asigna a cada video una de cuatro clases, definidas por los autores del dataset según el tipo de contenido y su público objetivo:

- **Suitable (apto):** contenido apropiado para niños de 1 a 5 años y relevante para sus intereses (caricaturas, canciones infantiles, videos educativos).
- **Disturbing (perturbador):** videos dirigidos a este mismo público, pero que contienen elementos inapropiados (insinuaciones sexuales, lenguaje abusivo, violencia gráfica, escenas de miedo, etc.); es el caso central del fenómeno Elsagate.
- **Restricted (restringido):** contenido no dirigido a niños pequeños, pero inapropiado para menores de 17 años en general (violencia explícita, lenguaje inapropiado, contenido sexual, consumo de drogas o alcohol).
- **Irrelevant (no relevante):** contenido apropiado pero no dirigido a niños pequeños (por ejemplo, videos de videojuegos o música dirigidos a públicos mayores).

Esta clasificación no equivale directamente a una escala binaria "apto/no apto": `suitable` e `irrelevant` son ambos apropiados (solo difieren en el público al que apuntan), mientras que `disturbing` y `restricted` representan contenido problemático. La distribución de clases, obtenida mediante anotación manual por revisores humanos, se presenta en la Sección 4.1. Esta variable fue elegida como objetivo porque es la única etiqueta verificada por humanos en el dataset (a diferencia de `prediction`, que proviene de un clasificador externo) y porque responde directamente a la pregunta de investigación central del proyecto.

---

## 3. Metodología

### 3.1 Enfoque del Análisis

El EDA se dividió en dos partes principales:

1. **Análisis Completo del Dataset de Entrenamiento:** Se realizó un análisis exhaustivo del dataset `dataset_groundtruth.csv` que contiene las etiquetas manuales confiables necesarias para el entrenamiento del modelo.

2. **Análisis Breve de Datasets Adicionales:** Se analizaron las primeras 1,000 líneas del primer archivo de cada dataset adicional (no corresponden a muestras aleatorias) para comprender su estructura y la distribución de sus predicciones automáticas.

### 3.2 Herramientas Utilizadas

- **Python 3.x** como lenguaje de programación principal
- **Pandas** para manipulación y análisis de datos
- **Matplotlib y Seaborn** para visualización de datos
- **Jupyter Notebook** como entorno de análisis interactivo

---

## 4. Análisis del Dataset de Entrenamiento (Ground Truth)

### 4.1 Distribución de Clases

El dataset de entrenamiento contiene 4,797 videos distribuidos en 4 categorías:

![Distribución de Clases](images/distrib_clases_groundtruth.png)

**Tabla 1: Distribución de Clases en Ground Truth**

| Categoría | Cantidad | Porcentaje |
|-----------|----------|------------|
| Irrelevant | 1,936 | 40.4% |
| Suitable | 1,513 | 31.5% |
| Disturbing | 929 | 19.4% |
| Restricted | 419 | 8.7% |
| **Total** | **4,797** | **100%** |

**Observaciones:**
- La clase mayoritaria es "irrelevant" (40.4%), seguida de "suitable" (31.5%)
- Las clases de interés para detección ("disturbing" y "restricted") representan el 28.1% del dataset
- Existe un desbalance de clases que deberá considerarse en el entrenamiento del modelo

### 4.2 Estadísticas Descriptivas

**Tabla 2: Estadísticas Descriptivas de Variables Numéricas**

| Variable | Datos válidos | Media | Desv. estándar | Mínimo | P25 | Mediana | P75 | Máximo |
|----------|---------------:|------:|---------------:|-------:|----:|--------:|----:|-------:|
| view_count | 4,794 | 6,142,716.96 | 43,836,538.05 | 0 | 1,140 | 90,093.50 | 1,775,452.75 | 1,664,660,083 |
| like_count | 4,669 | 26,878.13 | 155,478.81 | 0 | 10 | 502.00 | 10,122 | 4,392,637 |
| dislike_count | 4,669 | 3,718.92 | 25,425.08 | 0 | 1 | 53.00 | 1,097 | 1,071,513 |
| comment_count | 4,569 | 2,378.41 | 12,849.85 | 0 | 1 | 49.00 | 763 | 323,267 |
| duration_seconds | 4,796 | 752.06 | 1,986.99 | 0 | 167 | 345.50 | 706.25 | 39,433 |

**Observaciones:**
- Las métricas de engagement presentan alta variabilidad y valores ausentes, especialmente `comment_count` (228 faltantes) y `like_count`/`dislike_count` (128 faltantes cada uno)
- La duración promedio es de 752.06 segundos (12.5 minutos), mientras que la mediana es de 345.5 segundos (5.8 minutos)
- Los máximos muestran outliers significativos, sobre todo en visualizaciones y duración
- Se identificaron 11 videos con 0 vistas y 10 videos con duración 0, lo que exigirá tratar la división por cero al construir ratios de engagement en el preprocesamiento.

### 4.3 Análisis de Engagement por Categoría

![Distribución de Engagement por Categoría](images/distrib_engagement_clases_groundtruth.png)

> **Nota sobre la escala:** En escala logarítmica, los valores iguales a 0 no se representan gráficamente (11 videos con 0 vistas, 468 con 0 likes, 996 con 0 comentarios y 10 con duración 0).

**Tabla 3: `view_count` por categoría**

| Categoría | Datos válidos | Media | Mediana | Desv. estándar | Mínimo | Máximo |
|-----------|---------------:|------:|--------:|---------------:|-------:|-------:|
| Disturbing | 929 | 3,193,286.59 | 39,144 | 15,452,148.11 | 0 | 341,629,926 |
| Irrelevant | 1,934 | 3,486,723.52 | 10,436 | 29,164,201.75 | 0 | 699,895,118 |
| Restricted | 419 | 8,272,591.13 | 1,348,394 | 21,829,769.40 | 1 | 194,569,144 |
| Suitable | 1,512 | 10,761,959.60 | 407,330 | 68,498,396.60 | 0 | 1,664,660,083 |

**Tabla 3.1: `like_count` por categoría**

| Categoría | Datos válidos | Media | Mediana | Desv. estándar | Mínimo | Máximo |
|-----------|---------------:|------:|--------:|---------------:|-------:|-------:|
| Disturbing | 921 | 18,536.99 | 202 | 82,705.15 | 0 | 1,403,940 |
| Irrelevant | 1,908 | 35,003.91 | 87 | 210,949.86 | 0 | 4,392,637 |
| Restricted | 415 | 25,398.39 | 5,130 | 55,684.65 | 0 | 394,991 |
| Suitable | 1,425 | 21,820.08 | 1,342 | 119,035.69 | 0 | 3,228,147 |

**Tabla 3.2: `comment_count` por categoría**

| Categoría | Datos válidos | Media | Mediana | Desv. estándar | Mínimo | Máximo |
|-----------|---------------:|------:|--------:|---------------:|-------:|-------:|
| Disturbing | 904 | 1,705.64 | 27.0 | 7,041.60 | 0 | 111,492 |
| Irrelevant | 1,867 | 3,170.56 | 14.0 | 16,640.42 | 0 | 323,267 |
| Restricted | 410 | 3,379.91 | 531.0 | 8,173.27 | 0 | 86,011 |
| Suitable | 1,388 | 1,455.22 | 69.5 | 10,816.97 | 0 | 269,473 |

**Observaciones:**
- Los videos "suitable" tienen la mayor media de visualizaciones (10,761,959.60), mientras que "restricted" tiene la mayor mediana de visualizaciones
- La media de likes es mayor en "irrelevant" y la media de comentarios en "restricted"
- Las medias están fuertemente afectadas por outliers; por ello, las medianas ofrecen una comparación más robusta entre categorías

### 4.4 Categorías de YouTube

![Distribución de Categorías por Etiqueta](images/distrib_categ_etiqueta_groundtruth.png)

**Tabla 3.3: Distribución de las 17 categorías de YouTube en el dataset**

| ID | Categoría | Número de videos | % del total |
|---:|:---|---:|---:|
| 24 | Entertainment | 1,223 | 25.5% |
| 22 | People & Blogs | 1,088 | 22.7% |
| 1 | Film & Animation | 815 | 17.0% |
| 20 | Gaming | 365 | 7.6% |
| 27 | Education | 291 | 6.1% |
| 10 | Music | 266 | 5.5% |
| 23 | Comedy | 220 | 4.6% |
| 26 | Howto & Style | 114 | 2.4% |
| 25 | News & Politics | 106 | 2.2% |
| 17 | Sports | 97 | 2.0% |
| 2 | Autos & Vehicles | 55 | 1.1% |
| 28 | Science & Technology | 53 | 1.1% |
| 19 | Travel & Events | 39 | 0.8% |
| 15 | Pets & Animals | 29 | 0.6% |
| 29 | Nonprofits & Activism | 25 | 0.5% |
| 43 | Shows | 10 | 0.2% |
| 44 | Trailers | 1 | <0.1% |
| **Total** | | **4,796** | **100.0%** |

**Análisis:**
- Las tres categorías más frecuentes de la YouTube Data API son Entertainment (1,223 videos; 25.5%), People & Blogs (1,088 videos; 22.7%) y Film & Animation (815 videos; 17.0%), concentrando en conjunto el 65.2% del dataset.
- Se observa una diversidad de 17 categorías oficiales en total, abarcando desde canales de entretenimiento masivo hasta categorías de muy baja frecuencia como Shows (10 videos) y Trailers (1 video).
- Algunas categorías predominan en ciertas clases de clasificación: por ejemplo, Film & Animation concentra una alta proporción de videos clasificados como "suitable", mientras que Entertainment y People & Blogs se distribuyen de forma heterogénea entre todas las clases.

### 4.5 Análisis de Duración de Videos

![Distribución de Duración](images/distrib_duracion_groundtruth.png)

> **Nota sobre los límites de los gráficos:** El histograma está truncado a 60 minutos (135 videos superan este límite y no se muestran); el boxplot de la derecha tiene el eje Y limitado a 45 minutos (192 videos superan los 45 minutos y no aparecen en el gráfico), y los 10 videos con duración 0 tampoco se grafican en la escala logarítmica.

**Tabla 4: Estadísticas de Duración por Categoría (segundos)**

| Categoría | Datos válidos | Media | Mediana | Desviación estándar | Mínimo | Máximo |
|-----------|---------------:|------:|--------:|-------------------:|-------:|-------:|
| Disturbing | 929 | 508.71 | 359.0 | 656.71 | 4 | 14,959 |
| Irrelevant | 1,936 | 785.71 | 264.0 | 2,401.49 | 0 | 39,433 |
| Restricted | 419 | 703.29 | 350.0 | 1,471.53 | 9 | 20,826 |
| Suitable | 1,512 | 872.00 | 443.5 | 2,055.90 | 0 | 36,524 |

**Observaciones:**
- Los videos "suitable" tienen la mayor duración media (872.00 segundos) y mediana (443.5 segundos)
- Las medianas se sitúan entre 264 y 443.5 segundos (4.4-7.4 minutos)
- La variabilidad es especialmente alta en "irrelevant" y "suitable" por la presencia de videos muy largos

### 4.6 Calidad de Video

![Distribución de Definición de Video](images/distrib_definicion_groundtruth.png)

**Tabla 5: Distribución de Calidad de Video**

| Definición | Cantidad | Porcentaje (sobre 4,797) |
|------------|---------:|-------------------------:|
| hd | 3,748 | 78.1% |
| sd | 1,048 | 21.8% |

**Observaciones:**
- La mayoría de los videos (78.1%) están en alta definición (HD)
- La calidad del video no parece estar directamente correlacionada con la clasificación de contenido: la asociación entre definición y clase es estadísticamente significativa ($\chi^2 = 23.30$, $p = 3.5 \times 10^{-5}$), pero la fuerza de asociación es muy débil ($V \text{ de Cramér} = 0.07$).
- Hay un registro sin definición (`value_counts` suma 4,796); los porcentajes del notebook se calculan sobre el total de 4,797 videos

**Tabla 5.1: Definición por categoría**

| Definición | Disturbing | Irrelevant | Restricted | Suitable |
|------------|----------:|-----------:|-----------:|---------:|
| hd | 771 | 1,456 | 326 | 1,195 |
| sd | 158 | 480 | 93 | 317 |

### 4.7 Análisis de Tags

![Tags Más Frecuentes](images/tags_frecuentes_groundtruth.png)

Se identificaron **3,914 videos con tags** (883 videos, el 18.4%, no registran tags), agrupados en 3,705 listas distintas. En el procesamiento limpio de etiquetas individuales (separando por `|`, eliminando espacios en blanco y descartando cadenas vacías), se registran **40,096 tags únicos** y **73,640 asignaciones totales**, con un promedio de 18.8 tags por video con metadatos. Al normalizar a minúsculas (*case-folding*, estándar esencial para el procesamiento de texto y NLP), se consolidan **36,750 tags únicos** y un total de **73,084 asignaciones deduplicadas por video** (reduciendo repeticiones del mismo tag dentro de un video).

**Tabla 6: Top 20 tags individuales más frecuentes (normalizados a minúsculas) y distribución por clase**

| Posición | Tag | Frecuencia | % Disturbing | % Irrelevant | % Restricted | % Suitable |
|:---:|:---|---:|---:|---:|---:|---:|
| 1 | kids | 412 | 31.1% | 4.6% | 1.2% | 63.1% |
| 2 | funny | 400 | 39.5% | 19.0% | 6.5% | 35.0% |
| 3 | animation | 347 | 32.9% | 4.3% | 0.9% | 62.0% |
| 4 | cartoon | 299 | 24.7% | 2.7% | 0.3% | 72.2% |
| 5 | fun | 252 | 32.5% | 7.9% | 4.0% | 55.6% |
| 6 | toys | 221 | 24.9% | 4.5% | 0.5% | 70.1% |
| 7 | children | 215 | 20.9% | 7.9% | 0.0% | 71.2% |
| 8 | spiderman | 209 | 74.6% | 4.3% | 1.9% | 19.1% |
| 9 | elsa | 200 | 60.0% | 6.0% | 3.0% | 31.0% |
| 10 | disney | 195 | 30.8% | 5.6% | 1.0% | 62.6% |
| 11 | frozen | 194 | 57.7% | 5.7% | 2.6% | 34.0% |
| 12 | comedy | 186 | 24.2% | 23.7% | 10.2% | 41.9% |
| 13 | nursery rhymes | 165 | 31.5% | 0.0% | 0.0% | 68.5% |
| 14 | superhero | 156 | 74.4% | 6.4% | 0.6% | 18.6% |
| 15 | family | 147 | 23.1% | 17.0% | 2.0% | 57.8% |
| 16 | baby | 138 | 42.8% | 5.8% | 2.9% | 48.6% |
| 17 | for kids | 136 | 12.5% | 4.4% | 2.9% | 80.1% |
| 18 | cartoons | 136 | 25.7% | 2.9% | 0.7% | 70.6% |
| 19 | toy | 124 | 20.2% | 8.9% | 0.8% | 70.2% |
| 20 | compilation | 123 | 49.6% | 12.2% | 5.7% | 32.5% |

**Observaciones:**
- Al normalizar a minúsculas, se integran variantes que antes aparecían fragmentadas (como `cartoon` y `animation`) y se observa una marcada diferenciación entre categorías: tags de personajes específicos como `spiderman` (74.6%), `superhero` (74.4%), `elsa` (60.0%) y `frozen` (57.7%) coinciden con una alta proporción de videos en `disturbing`.
- Estos tags asociados a contenido perturbador no provienen de un canal aislado, sino que se distribuyen ampliamente entre numerosos creadores (181 canales distintos para `spiderman`, 171 para `elsa`, 169 para `frozen` y 133 para `superhero`), donde el canal con mayor número de videos de ese tag aporta menos del 4% del total.
- En contraste, tags genéricos como `for kids` (80.1%), `cartoon` (72.2%), `children` (71.2%), `cartoons` (70.6%), `toys`/`toy` (~70%) y `nursery rhymes` (68.5%) se asocian predominantemente con la clase `suitable`.
- Se detectan casos puntuales de concentración en etiquetas generales: para `children` (215 videos en 129 canales) y `kids` (412 videos en 270 canales), un único canal (`UC9-oOzriEPfpPvjTOAu3Cww`) reúne el 14.4% (31 videos) y el 7.3% (30 videos) de las observaciones, respectivamente.
- La presencia de tags individuales normalizados constituye un insumo discriminante potencialmente relevante para los modelos de procesamiento de lenguaje natural (NLP) en la Etapa 2.

### 4.8 Análisis de Títulos

![Análisis de Longitud de Títulos](images/distrib_titulo_groundtruth.png)

**Estadísticas generales de títulos**

| Medida | Longitud del título | Palabras del título |
|--------|--------------------:|--------------------:|
| Datos válidos | 4,797 | 4,797 |
| Media | 53.74 | 9.34 |
| Desviación estándar | 24.92 | 4.37 |
| Mínimo | 1 | 1 |
| Primer cuartil | 34 | 6 |
| Mediana | 51 | 9 |
| Tercer cuartil | 73 | 13 |
| Máximo | 113 | 22 |

**Tabla 7: Estadísticas de títulos por categoría**

| Categoría | Longitud media | Longitud mediana | Desv. estándar | Palabras media | Palabras mediana | Desv. estándar |
|-----------|---------------:|-----------------:|---------------:|---------------:|-----------------:|---------------:|
| Disturbing | 58.98 | 56.0 | 28.18 | 10.04 | 10.0 | 4.55 |
| Irrelevant | 46.80 | 43.5 | 22.82 | 8.12 | 8.0 | 4.17 |
| Restricted | 44.79 | 42.0 | 20.88 | 7.79 | 7.0 | 3.70 |
| Suitable | 61.88 | 61.0 | 22.98 | 10.91 | 11.0 | 4.05 |

**Observaciones:**
- Los títulos de videos "suitable" tienen la mayor longitud media (61.88 caracteres) y cantidad media de palabras (10.91); "disturbing" ocupa el segundo lugar
- La longitud del título podría ser un feature discriminante
- "restricted" presenta los títulos más cortos en promedio, mientras que "irrelevant" tiene más observaciones

### 4.9 Detección de Outliers

Se aplicó el método del rango intercuartílico (IQR) sobre las cinco variables numéricas (`view_count`, `like_count`, `dislike_count`, `comment_count`, `duration_seconds`), tanto en su escala original como sobre una transformación logarítmica (`log1p`). Esta segunda versión es necesaria porque, dada la fuerte asimetría de las métricas de engagement (ver Tabla 2), el IQR calculado en escala original marca como "outlier" a una fracción muy grande de las observaciones, lo cual no es informativo.

![Boxplots de Outliers](images/outliers_boxplots_groundtruth.png)

**Tabla 12: Outliers detectados por variable (método IQR)**

| Variable | N° Outliers (escala original) | % Outliers (original) | N° Outliers (escala log) | % Outliers (log) |
|----------|-------------------------------:|-----------------------:|---------------------------:|-------------------:|
| view_count | 802 | 16.73% | 0 | 0.00% |
| like_count | 730 | 15.64% | 0 | 0.00% |
| dislike_count | 791 | 16.94% | 0 | 0.00% |
| comment_count | 721 | 15.78% | 0 | 0.00% |
| duration_seconds | 385 | 8.03% | 173 | 3.61% |

**Observaciones:**
- En escala original, el IQR detecta una alta cantidad de "outliers" en las variables de engagement (entre 15.6% y 16.9%, superando los 700 videos en cada una) debido a la asimetría extrema y cola pesada típica de YouTube (ver máximos en Tabla 2).
- En escala logarítmica (`log1p`), el número de outliers en las cuatro variables de engagement desciende a **0 (0.00%)**, demostrando que las observaciones siguen una distribución log-normal coherente y no constituyen errores ni datos anómalos. Únicamente `duration_seconds` retiene 173 outliers (3.61%) correspondientes a videos notablemente cortos o largos.
- Al cruzar los valores extremos de `view_count` (> 4.4M) y `like_count` (> 25.3K) con `classification_label`, se observa que están repartidos entre todas las clases (en vistas: 379 suitable, 175 irrelevant, 132 restricted y 116 disturbing; en likes: 300 irrelevant, 214 suitable, 121 disturbing y 95 restricted). Por ende, corresponden a contenido popular legítimo dentro de cada categoría y no deben descartarse, sino normalizarse o transformarse mediante log1p en la Etapa 2.

### 4.10 Cuantificación del Desbalance de Clases

Complementando la distribución de clases de la Sección 4.1, se calculó el ratio de desbalance (*imbalance ratio*, IR) de cada clase respecto a la clase mayoritaria (`irrelevant`).

**Tabla 13: Ratio de desbalance por clase**

| Categoría | Cantidad | IR respecto a "irrelevant" |
|-----------|---------:|----------------------------:|
| Irrelevant | 1,936 | 1.00x |
| Suitable | 1,513 | 1.28x |
| Disturbing | 929 | 2.08x |
| Restricted | 419 | 4.62x |

**Observaciones:**
- La clase `restricted`, la más relevante para detectar contenido de mayor riesgo, es 4.6 veces más pequeña que la clase mayoritaria.
- Este nivel de desbalance (IR > 4x en la clase minoritaria) justifica el uso de técnicas específicas en la Etapa 2: class weights, sobremuestreo (SMOTE) o una combinación de ambas, tal como se planteó en la propuesta de próximos pasos.

### 4.11 Relaciones entre Variables (Análisis de Correlación)

Se analizó la correlación entre las variables numéricas (Pearson en escala original, Pearson en escala log1p y Spearman) para identificar redundancia y posibles problemas de multicolinealidad de cara al feature engineering de la Etapa 2. Adicionalmente, se aplicó la prueba de Kruskal-Wallis para evaluar si la distribución de cada variable numérica difiere de forma estadísticamente significativa entre las cuatro categorías de `classification_label`.

![Matriz de Correlación](images/correlacion_variables_groundtruth.png)

**Tabla 14: Resultados de la prueba de Kruskal-Wallis (variable numérica vs. clase)**

| Variable | Estadístico H | p-value | ¿Diferencia significativa entre clases? |
|----------|---------------:|--------:|:----------------------------------------|
| view_count | 523.87 | 3.20e-113 | Sí (p < 0.05) |
| like_count | 232.61 | 3.76e-50 | Sí (p < 0.05) |
| dislike_count | 399.39 | 3.00e-86 | Sí (p < 0.05) |
| comment_count | 150.65 | 1.91e-32 | Sí (p < 0.05) |
| duration_seconds | 129.94 | 5.57e-28 | Sí (p < 0.05) |

**Observaciones:**
- Las métricas de engagement presentan una correlación excepcionalmente alta entre sí (especialmente en Spearman y escala logarítmica): `view_count` y `dislike_count` (Spearman = 0.966, Pearson log = 0.940), `view_count` y `like_count` (Spearman = 0.956, Pearson log = 0.955), y `like_count` y `comment_count` (Spearman = 0.948, Pearson log = 0.947). Esta colinealidad casi unitaria desaconseja emplear los cuatro conteos como variables directas simultáneas; en su lugar, se deben formular ratios normalizados (likes/views, comments/views) según lo planificado para el feature engineering.
- Por su parte, `duration_seconds` mantiene correlaciones bajas con todas las variables de engagement (Pearson log entre 0.228 y 0.241, Spearman entre 0.217 y 0.225), por lo que provee una señal complementaria e independiente.
- En la prueba no paramétrica de Kruskal-Wallis, **las 5 variables numéricas** resultaron estadísticamente significativas ($p < 0.05$, de hecho todas con $p < 10^{-27}$), indicando que sus distribuciones difieren entre las cuatro clases. No obstante, dado el tamaño muestral amplio (~4,800 observaciones), diferencias distributivas pequeñas pueden alcanzar significancia estadística con facilidad; por ello, este resultado funciona como un filtro preliminar positivo (ninguna variable se descarta a priori), debiendo confirmarse su verdadera capacidad predictiva en la Etapa 2 mediante métricas de importancia de features con los modelos entrenados.

---

## 5. Análisis de Datasets Adicionales (Primeras 1,000 líneas)

> [!NOTE]
> Las muestras analizadas corresponden a las primeras 1,000 líneas del primer archivo comprimido de cada conjunto, no a una muestra aleatoria. En ellas predominan videos que también están presentes en el ground truth (780 en `elsagate_related`, 70 en `other_child_related`, 34 en `random_videos` y 490 en `popular_videos`), por lo que los porcentajes de videos inapropiados no representan la prevalencia real en los datasets completos. En `elsagate_related`, 472 de los 780 videos pertenecientes al ground truth (60.5%) fueron predichos como inapropiados, frente a solo 1 de los 220 videos restantes no pertenecientes al ground truth (0.5%). En `other_child_related` (15), `random_videos` (3) y `popular_videos` (27), la totalidad de los videos predichos como inapropiados pertenecen al ground truth.

### 5.1 Elsagate Related Videos (Primeras 1,000 líneas)

![Distribución de Predicciones - Elsagate](images/distrib_elsagate.png)

**Tabla 8: Distribución de Predicciones en Elsagate**

| Predicción | Cantidad | Porcentaje |
|------------|----------|------------|
| appropriate | 527 | 52.7% |
| inappropriate | 473 | 47.3% |

En la muestra, 780 videos también están en el ground truth.

**Observaciones:**
- Casi la mitad de los videos en esta muestra específica (47.3%) fueron clasificados como inapropiados, concentrándose casi en su totalidad (472 de 473) en los videos que provienen del ground truth.
- Esta muestra representa solo las primeras 1,000 líneas de un archivo y no una muestra aleatoria del dataset completo de ~233K videos.

### 5.2 Other Child Related Videos (Primeras 1,000 líneas)

![Distribución de Predicciones - Other Child](images/distrib_other_child.png)

**Tabla 9: Distribución de Predicciones en Other Child**

| Predicción | Cantidad | Porcentaje |
|------------|----------|------------|
| appropriate | 985 | 98.5% |
| inappropriate | 15 | 1.5% |

En la muestra, 70 videos también están en el ground truth.

**Observaciones:**
- La mayoría de los videos en la muestra fueron clasificados como apropiados (98.5%).
- Los 15 videos clasificados como inapropiados corresponden a videos que pertenecen al ground truth.
- Baja proporción observada de contenido inapropiado en esta muestra.

### 5.3 Random Videos (Primeras 1,000 líneas)

![Distribución de Predicciones - Random](images/distrib_random.png)

**Tabla 10: Distribución de Predicciones en Random**

| Predicción | Cantidad | Porcentaje |
|------------|----------|------------|
| appropriate | 997 | 99.7% |
| inappropriate | 3 | 0.3% |

En la muestra, 34 videos también están en el ground truth.

**Observaciones:**
- La proporción de contenido inapropiado es muy baja en esta muestra (0.3%, correspondiente a 3 videos del ground truth).
- El resultado debe interpretarse con cautela porque se trata de las primeras 1,000 líneas y no de una muestra aleatoria.
- La generalización y eficacia del modelo se evaluará formalmente sobre el conjunto de prueba reservado del ground truth; estos datos adicionales sin etiquetar servirán para comprobar la consistencia de predicciones en inferencia.

### 5.4 Popular Videos (Primeras 1,000 líneas)

![Distribución de Predicciones - Popular](images/distrib_popular.png)

**Tabla 11: Distribución de Predicciones en Popular**

| Predicción | Cantidad | Porcentaje |
|------------|----------|------------|
| appropriate | 973 | 97.3% |
| inappropriate | 27 | 2.7% |

En la muestra, 490 videos también están en el ground truth.

**Observaciones:**
- Los videos populares en la muestra resultan predominantemente clasificados como apropiados (97.3%).
- Los 27 videos clasificados como inapropiados corresponden en su totalidad a videos que forman parte del ground truth.
- Baja proporción observada de contenido predicho como inapropiado en esta muestra.

---

## 6. Análisis de Posible Fuga de Datos (Data Leakage)

Se revisaron cuatro fuentes potenciales de fuga de datos relevantes para el diseño experimental de la Etapa 2:

### 6.1 Columnas que codifican la etiqueta o la predicción original

El dataset incluye las columnas `prediction` e `is_ground_truth`, generadas por el clasificador automático de los autores del dataset y por el proceso de anotación, respectivamente. **Ninguna de las dos debe usarse como feature**: no son información disponible de forma independiente para un video nuevo, sino subproductos del propio proceso de etiquetado/clasificación original. Se verificó su presencia y valores en `dataset_groundtruth.csv` mediante el notebook: `prediction` contiene un 100% de valores nulos (4,797 registros NaN) e `is_ground_truth` toma el valor constante 1 para los 4,797 registros.

### 6.2 Duplicados dentro del ground truth

Se verificó la existencia de `video_id` duplicados dentro de `dataset_groundtruth.csv`. De existir duplicados, deben resolverse antes de hacer el split train/validation/test (quedarse con un único registro por video), ya que un mismo video repetido en train y en validation infla artificialmente el desempeño reportado.

Tras la ejecución del notebook, se constató que existen **0 videos duplicados** (mismo `video_id`) dentro del dataset de entrenamiento. Todos los 4,797 registros representan videos únicos.

### 6.3 Solapamiento entre ground truth y datasets adicionales

Se comparó el conjunto de `video_id` de `dataset_groundtruth.csv` contra los `video_id` presentes en las muestras de `elsagate_related`, `other_child_related`, `random_videos` y `popular_videos`. Esto es relevante porque el plan de trabajo (Sección de "Próximos pasos") usa el ground truth para entrenar y estos datasets adicionales para comprobación de consistencia y predicciones con el modelo entrenado (sin etiquetas humanas, salvo los videos que ya están en el ground truth): si un mismo video aparece en ambos conjuntos, cualquier comprobación sobre ese video dejaría de ser una prueba genuina de generalización.

**Tabla 15: Videos en común entre ground truth y cada dataset adicional (sobre la muestra de 1,000)**

| Dataset adicional | Videos en común con ground truth |
|--------------------|-----------------------------------:|
| elsagate_related | 780 (78.0%) |
| other_child_related | 70 (7.0%) |
| random_videos | 34 (3.4%) |
| popular_videos | 490 (49.0%) |

### 6.4 Concentración de videos por canal

Se analizó cuántos videos del ground truth provienen del mismo `channel_id`, y si los canales con más videos tienden a concentrarse en una sola clase. Si esto ocurre, existe riesgo de que un modelo "memorice" características propias del canal (estilo de miniatura, patrones de título recurrentes del mismo uploader) en lugar de aprender patrones generalizables de contenido inapropiado — y ese riesgo se convierte en fuga de datos real si videos del mismo canal quedan repartidos entre train y validation/test.

Al ejecutar el análisis, se identificaron 3,818 canales únicos (por `channel_id`; 3,811 por `channel_title`), de los cuales **418 canales tienen más de un video** en el ground truth (con un máximo de 33 videos para un canal). Al inspeccionar la distribución de clases en los canales con mayor presencia, existen 13 canales con 12 o más videos (debido a que 6 canales empatan con exactamente 12 videos): **11 concentran el 100% en `suitable`**, 1 tiene el 91.7% en `suitable` (11 suitable y 1 disturbing, `UCBzZ9lJmcsPRbiyHPbLYvOA`) y solo 1 es mixto (20 disturbing y 13 suitable, `UC9-oOzriEPfpPvjTOAu3Cww`). Cualquiera sea el criterio de desempate utilizado para seleccionar un 'top 10', entre 8 y 9 de los 10 canales concentran el 100% en una sola clase. Esto demuestra una fuerte concentración por canal y ratifica el riesgo de sobreajuste/fuga si no se controla adecuadamente.

### 6.5 Recomendaciones derivadas

- Excluir `prediction` e `is_ground_truth` de las features de entrenamiento.
- Eliminar duplicados de `video_id` en el ground truth antes del split, si los hubiera.
- Excluir de la comprobación en datasets adicionales cualquier video cuyo `video_id` ya esté en el ground truth de entrenamiento.
- Si hay concentración fuerte de clase por canal, usar un split agrupado por canal con `StratifiedGroupKFold` (tanto para reservar el conjunto de prueba como para la validación cruzada), además de la estratificación por clase planteada en el plan de preprocesamiento, para que videos de un mismo canal no queden repartidos entre train y validation/test.
- Usar `channel_id` exclusivamente para definir los grupos de validación y excluir `channel_id`, `channel_title` y `video_id` como variables predictoras (features).
- Ajustar todas las transformaciones (imputación, escalamiento, TF-IDF y balanceo con SMOTE) estrictamente solo con los datos de entrenamiento para prevenir fuga de información (*data leakage*).

---

## 7. Conclusiones y Recomendaciones

### 7.1 Hallazgos Principales

1. **Balance de Clases:** El dataset de entrenamiento presenta un desbalance de clases significativo, con "irrelevant" siendo la clase mayoritaria (1,936 videos; 40.4%). Esto deberá abordarse mediante técnicas de balanceo durante el entrenamiento.

2. **Engagement y Contenido Inapropiado:** "Suitable" presenta la mayor media de visualizaciones, mientras que "restricted" presenta la mayor mediana de visualizaciones y media de comentarios. Las distribuciones tienen outliers fuertes, por lo que no es correcto atribuir el mayor engagement general a "disturbing".

3. **Características Discriminantes:** 
   - La duración: "suitable" tiene la mayor media (872.00 segundos) y mediana (443.5 segundos)
   - Longitud de títulos: "suitable" tiene los títulos más largos en promedio (61.88 caracteres)
   - Tags específicos muestran asociaciones dispares con las clases: etiquetas normalizadas como `spiderman` (74.6% disturbing), `superhero` (74.4%), `elsa` (60.0%) y `frozen` (57.7%) coinciden marcadamente con contenido perturbador a lo largo de más de 100 canales distintos, mientras que tags genéricos como `for kids` (80.1% suitable), `cartoon` (72.2%), `children` (71.2%) y `nursery rhymes` (68.5%) coinciden con contenido adecuado.

4. **Distribución en Datasets Adicionales:** Con las muestras actuales (primeras 1,000 líneas) no es posible estimar la prevalencia de contenido inapropiado en los datasets adicionales: los videos predichos como inapropiados provienen del ground truth (472 de 473 en `elsagate_related` y la totalidad en los otros tres conjuntos).

### 7.2 Resumen numérico del dataset de entrenamiento

- **Videos:** 4,797; **vistas totales:** 29,448,185,127
- **Media de vistas:** 6,142,716.96; **media de likes:** 26,878.13; **media de comentarios:** 2,378.41
- **Duración:** media de 752.06 segundos (12.5 minutos) y mediana de 345.5 segundos (5.8 minutos)
- **Título:** media de 53.7 caracteres y 9.3 palabras
- **Diversidad:** 17 categorías de YouTube, 3,818 canales (por `channel_id`; 3,811 por `channel_title`), 40,096 tags individuales limpios (36,750 normalizados a minúsculas; 3,705 listas distintas)
- **Definición:** 3,748 videos HD (78.1%) y 1,048 SD (21.8%); un registro no tiene definición

### 7.3 Recomendaciones para el Modelado

1. **Preprocesamiento:**
   - Implementar técnicas de balanceo de clases (SMOTE, class weights, etc.)
   - Normalizar variables numéricas con alta variabilidad
   - Considerar transformación logarítmica para métricas de engagement

2. **Feature Engineering:**
   - Crear features basados en análisis de texto de títulos y descripciones
   - Utilizar embeddings de tags para capturar similitudes semánticas
   - Incluir ratios de engagement (likes/views, comments/views)

3. **Validación:**
   - Utilizar estratificación por clase y agrupación por canal en el split train/test y en la validación cruzada
   - Considerar métricas específicas para clases minoritarias (F1-score, recall)
   - Implementar validación cruzada robusta

4. **Despliegue (escenario hipotético):**
   - Establecer umbrales de probabilidad según el riesgo tolerable
   - Considerar sistema de revisión humana para casos de baja confianza
   - Monitorear continuamente el performance en datos reales

### 7.4 Limitaciones del Análisis

1. **Muestra vs Población:** Los datasets adicionales se analizaron solo con las primeras 1,000 líneas de un archivo de cada conjunto, las cuales no corresponden a muestras aleatorias y presentan una alta concentración de videos del ground truth, por lo que no representan la distribución real de los datasets completos.

2. **Temporalidad:** Los datos corresponden a un periodo específico (2018), por lo que pueden no reflejar patrones actuales de YouTube.

3. **Sesgo de Etiquetado:** Las etiquetas manuales pueden contener sesgos subjetivos de los anotadores humanos.

4. **Evolución de la Plataforma:** YouTube ha cambiado significativamente desde 2018, por lo que los patrones pueden haber evolucionado.

---

## 8. Próximos Pasos

1. **Feature Engineering:** Aplicar `log1p` a las métricas de engagement, crear ratios (likes/views, comments/views) e implementar técnicas de NLP para analizar títulos, descripciones y tags (normalizar tags a minúsculas, TF-IDF o embeddings). Se excluyen `prediction`, `is_ground_truth`, `video_id`, `channel_id` y `channel_title` como features.

2. **Modelado Baseline:** Definir un baseline de referencia (clasificador que predice siempre la clase mayoritaria) y comparar modelos que permitan aislar el aporte de cada tipo de información: por ejemplo, regresión logística con TF-IDF para el texto, un modelo de árboles (LightGBM o Random Forest) para los metadatos, y un modelo combinado de ambos. La selección final se justificará en la Etapa 2 según los resultados. Se abordará el desbalance de clases con class weights o SMOTE.

3. **Modelos Avanzados:** Explorar embeddings preentrenados o transformers para texto solo si el texto resulta ser la señal más fuerte en la etapa anterior. El uso de thumbnails (CNN) queda como extensión.

4. **Validación:** Reservar cerca del 20% del ground truth como conjunto de prueba, estratificado por clase y agrupado por canal (Sección 6.4), y aplicar validación cruzada con el mismo criterio sobre el resto. La evaluación se centrará en **F1 macro y por clase** —en lugar de exactitud (*accuracy*), la cual se vería inflada y distorsionada por el desbalance de hasta 4.6x a favor de la clase mayoritaria—, priorizando además el **recall en las clases críticas** (`restricted` y `disturbing`) para minimizar los falsos negativos en moderación infantil. Las transformaciones (imputación, TF-IDF, SMOTE) se ajustarán estrictamente solo con los datos de entrenamiento para prevenir fuga de datos (*data leakage*).

5. **Pruebas en datasets adicionales:** Aplicar el modelo final a los datasets adicionales, excluyendo los videos que ya están en el ground truth (Sección 6.3). Como no tienen etiquetas humanas, sirven como comprobación de consistencia (proporción de videos predichos como inapropiados y acuerdo con `prediction`), no para medir la eficacia, que se mide en el conjunto de prueba reservado.

---

## 9. Referencias

- Brandwatch. (2026). *23 YouTube stats and facts you should know*. https://www.brandwatch.com/blog/youtube-stats/
- Google. (s. f.). *YouTube Data API v3*. https://developers.google.com/youtube/v3
- Papadamou, K., Papasavva, A., Zannettou, S., Blackburn, J., Kourtellis, N., Leontiadis, I., Stringhini, G., & Sirivianos, M. (2020a). Disturbed YouTube for kids: Characterizing and detecting inappropriate videos targeting young children. *Proceedings of the International AAAI Conference on Web and Social Media, 14*(1), 522–533. https://doi.org/10.1609/icwsm.v14i1.7320
- Papadamou, K., Papasavva, A., Zannettou, S., Blackburn, J., Kourtellis, N., Leontiadis, I., Stringhini, G., & Sirivianos, M. (2020b). *Dataset for "Disturbed YouTube for kids: Characterizing and detecting inappropriate videos targeting young children"* [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.3632781
- United States Cybersecurity Magazine. (s. f.). *ElsaGate: The problem with algorithms*. https://www.uscybersecurity.net/elsagate/
- YouTube. (2020, 22 de septiembre). *Using technology to more consistently apply age restrictions*. YouTube Blog. https://blog.youtube/news-and-events/using-technology-more-consistently-apply-age-restrictions/