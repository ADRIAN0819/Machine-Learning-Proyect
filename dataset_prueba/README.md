# 📊 Extractor de Dataset de Estímulo Audiovisual (YouTube -> CSV)

Este repositorio contiene un pipeline modular de alto rendimiento en Python para realizar **scraping y análisis cuantitativo de estímulo visual y auditivo** a partir de videos de YouTube.

El objetivo principal es transformar secuencias de video en un **dataset estructurado tabular (`.csv`)** con variables numéricas listas para **Análisis Exploratorio de Datos (EDA)** y modelos de **Machine Learning** (Clustering/Agrupamiento como K-Means/DBSCAN, Reducción de Dimensionalidad como PCA/t-SNE, o Clasificación/Regresión de engagement/retención).

---

## ⚡ Optimizaciones de Rendimiento
- **Descarga Parcial Selectiva (`yt-dlp`)**: Solo descarga los primeros 120 segundos (2 minutos) del video en baja resolución (360p o 480p). Esto ahorra hasta un 90% de ancho de banda y almacenamiento en disco.
- **Procesamiento de Fotogramas Ultrarrápido (`OpenCV`)**: El cálculo del flujo óptico denso (Farneback) y de bordes (Canny) se ejecuta a resolución interna optimizada, alcanzando velocidades superiores a 60-120 FPS.
- **Gestión Automática de Almacenamiento**: Los archivos de video temporales se eliminan automáticamente tras procesarse para evitar saturar el disco.
- **Extracción de Audio Especializada (`Librosa`)**: Obtención directa de energía acústica, ritmo y envolventes espectrales.

---

## 🛠️ Instalación y Requisitos

1. Asegúrate de tener Python 3.10+ instalado.
2. Instala las dependencias del proyecto:

```bash
py -m pip install -r requirements.txt
```

*(No requiere configuración manual de FFmpeg: el script detecta e integra automáticamente el backend FFmpeg empaquetado).*

---

## 🚀 Guía de Uso

### 1. Extracción Automática de URLs Infantiles (Opcional)
Si deseas regenerar o actualizar las 100 URLs de videos infantiles (50 de Alta Estimulación vs 50 de Baja/Moderada Estimulación):

```bash
py extraccion_datos/fetch_kids_urls.py
```

### 2. Ejecución del Pipeline con archivo de URLs (Recomendado para lotes)
Procesa las 100 URLs presentes en `urls_ejemplo.txt`:

```bash
py extraccion_datos/extract_stimulus_dataset.py --urls-file extraccion_datos/urls_ejemplo.txt
```

### 3. Ejecución directa con URLs individuales
```bash
py extraccion_datos/extract_stimulus_dataset.py --urls https://www.youtube.com/watch?v=dQw4w9WgXcQ https://www.youtube.com/watch?v=jNQXAC9IVRw
```

### 3. Opciones y Parámetros Disponibles

| Parámetro | Abreviatura | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--urls-file` | `-f` | `urls_ejemplo.txt` | Ruta al archivo `.txt` con URLs a procesar |
| `--urls` | | `None` | Lista de URLs pasadas directamente por línea de comandos |
| `--output` | `-o` | `dataset_estimulacion.csv` | Ruta del dataset tabular consolidado a nivel de video |
| `--temporal-output`| `-t` | `dataset_estimulacion_temporal.csv` | Ruta del dataset con la serie temporal segundo a segundo |
| `--max-duration` | `-d` | `120` | Segundos máximos a descargar y analizar por video |
| `--quality` | `-q` | `360` | Resolución máxima de descarga (altura en píxeles: 360, 480) |
| `--keep-videos` | | `False` | Si se incluye, no borra los archivos `.mp4` temporales |
| `--verbose` | | `False` | Muestra logs de depuración detallados |

---

## 📋 Diccionario de Variables del Dataset

### 1. Dataset Consolidado (`dataset_estimulacion.csv` - 28 columnas)

| Columna | Tipo | Descripción y Significado para ML |
| :--- | :--- | :--- |
| `video_id` | `string` | Identificador único del video de YouTube |
| `titulo` | `string` | Título del video |
| `canal` | `string` | Nombre del canal / creador (10 canales independientes) |
| `categoria_estimulacion` | `string` | Etiqueta curada (`Alta Estimulacion` vs `Baja/Moderada Estimulacion`) |
| `tipo_produccion` | `string` | Variable de control estructural (`animacion` vs `accion_real`) |
| `formato` | `string` | Variable de control de contenido (`musical`, `narrativo`, `educativo_exploratorio`) |
| `fuente_etiqueta` | `string` | Origen de la etiqueta (`Curaduria_Canales`, `CLI`, etc.) |
| `url` | `string` | URL original analizada |
| `duracion_analizada_seg` | `float` | Segundos reales procesados del video (ej. hasta 120.0s) |
| `cortes_por_minuto` | `float` | Tasa de cambios bruscos de cámara/toma por minuto (**Pacing visual**) |
| `duracion_promedio_escena` | `float` | Duración media en segundos de cada toma antes de un corte |
| `magnitud_movimiento_promedio` | `float` | Promedio del flujo óptico de Farneback (velocidad de objetos/cámara) |
| `variabilidad_movimiento` | `float` | Desviación estándar del flujo óptico (picos de acción frenética) |
| `luminosidad_promedio` | `float` | Valor medio del canal V en HSV [0-255] (nivel de luz) |
| `variabilidad_luminosidad` | `float` | Desviación estándar temporal de la luminosidad entre fotogramas |
| `contraste_promedio` | `float` | Desviación estándar media del brillo intra-frame (rango dinámico) |
| `variabilidad_contraste` | `float` | Desviación estándar temporal del contraste a lo largo del tiempo |
| `saturacion_promedio` | `float` | Valor medio del canal S en HSV [0-255] (colores vivos/neón vs tenues) |
| `variabilidad_saturacion` | `float` | Desviación estándar temporal de la saturación entre fotogramas |
| `flicker_promedio` | `float` | Promedio de la diferencia absoluta de luminosidad frame a frame |
| `destellos_por_minuto` | `float` | Saltos bruscos de luminosidad (>= 15 pts) por minuto (**Flicker rate**) |
| `riqueza_cromatica` | `float` | Entropía del histograma de tonalidades H [0-100] (variedad de colores) |
| `densidad_bordes_promedio` | `float` | % de píxeles con bordes detectados vía Canny (complejidad/densidad espacial) |
| `energia_audio_promedio` | `float` | RMS (Root Mean Square) medio de la señal de audio (volumen/intensidad) |
| `variabilidad_audio` | `float` | Desviación estándar del RMS (contrastes entre silencio y explosiones/gritos) |
| `tempo_bpm` | `float` | Ritmo estimado de la pista de audio en pulsos por minuto (BPM) |
| `onsets_por_minuto` | `float` | Tasa de ataques/golpes/transitorios acústicos por minuto (**Onset rate**) |
| `brillo_espectral_promedio` | `float` | Centroide espectral en Hz (presencia de agudos, chillidos y efectos estridentes) |

### 2. Dataset Temporal (`dataset_estimulacion_temporal.csv`)
Muestra la evolución **segundo a segundo** de cada video para identificar patrones dinámicos (estimulación constante vs picos de estimulación / curvas de retención):

- `video_id`: Identificador del video.
- `second`: Segundo correspondiente (0, 1, 2, ..., 119).
- `cortes_segundo`: Cantidad de cortes de escena ocurridos en ese segundo específico.
- `motion`: Flujo óptico promedio en ese segundo.
- `luminosity`: Nivel de brillo medio en ese segundo.
- `contrast`: Contraste medio en ese segundo.
- `saturation`: Saturación media en ese segundo.
- `chroma_richness`: Riqueza cromática en ese segundo.
- `edge_density`: Densidad de bordes (%) en ese segundo.
- `audio_energy`: Energía RMS media del audio en ese segundo.
- `spectral_brightness`: Centroide espectral medio (Hz) en ese segundo.

---

## 📂 Estructura del Código

```text
Machine-Learning-Proyect/
├── README.md                      # Documentación del proyecto
├── requirements.txt               # Lista de dependencias del entorno
├── dataset_estimulacion.csv          # Dataset consolidado por video (28 variables)
├── dataset_estimulacion_temporal.csv # Serie temporal segundo a segundo
└── extraccion_datos/
    ├── extract_stimulus_dataset.py    # Punto de entrada CLI ejecutable
    ├── fetch_kids_urls.py             # Extractor automático de URLs infantiles
    ├── urls_ejemplo.txt               # 100 URLs curadas y categorizadas
    ├── requirements.txt               # Dependencias de la extracción
    └── src/
        ├── __init__.py
        ├── downloader.py              # Descarga selectiva con yt-dlp y ffmpeg
        ├── video_analyzer.py          # Extracción con OpenCV y PySceneDetect
        ├── audio_analyzer.py          # Extracción acústica con Librosa
        └── pipeline.py                # Coordinador del pipeline y exportación CSV
```
>>>>>>> 574d7f2 (feat: reorganizar pipeline en extraccion_datos, agregar variables de control y gitignore)
