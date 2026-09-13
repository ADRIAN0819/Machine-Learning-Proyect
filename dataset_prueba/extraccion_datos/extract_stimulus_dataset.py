"""
Script Principal para Extracción y Análisis Cuantitativo de Estímulo Audiovisual en Videos de YouTube.

Uso:
    py extract_stimulus_dataset.py --urls-file urls_ejemplo.txt
    py extract_stimulus_dataset.py --urls https://www.youtube.com/watch?v=XXXXX
"""

import os
import sys
import argparse
import logging

# Asegurar codificación UTF-8 en consola Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.pipeline import StimulusPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scraping y Análisis Cuantitativo de Estímulo Audiovisual en YouTube para Machine Learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--urls",
        nargs="+",
        help="Lista de URLs de YouTube separadas por espacio."
    )
    parser.add_argument(
        "-f", "--urls-file",
        type=str,
        default="urls_ejemplo.txt",
        help="Ruta a un archivo de texto con una URL por línea."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset_estimulacion.csv",
        help="Ruta del archivo CSV de salida con las métricas consolidadas por video."
    )
    parser.add_argument(
        "-t", "--temporal-output",
        type=str,
        default="dataset_estimulacion_temporal.csv",
        help="Ruta del archivo CSV de salida con la serie temporal segundo a segundo."
    )
    parser.add_argument(
        "-d", "--max-duration",
        type=int,
        default=120,
        help="Segundos máximos a descargar y analizar por video (por defecto 120 = primeros 2 min)."
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=360,
        help="Resolución máxima de descarga (altura en píxeles, ej. 360p o 480p)."
    )
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Si se activa, conserva los archivos de video descargados en 'temp_videos'."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra logs detallados de depuración."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    items_to_process = []

    # Priorizar URLs pasadas directamente por CLI
    if args.urls:
        for u in args.urls:
            items_to_process.append((u.strip(), "No especificado", "CLI"))
    else:
        urls_file_path = args.urls_file
        if not os.path.exists(urls_file_path):
            alt_path = os.path.join(script_dir, args.urls_file)
            if os.path.exists(alt_path):
                urls_file_path = alt_path

        if os.path.exists(urls_file_path):
            current_category = "No especificado"
            current_source = "Curaduria_Canales"

            with open(urls_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue

                    if line_str.startswith("#"):
                        line_lower = line_str.lower()
                        if "alta estimulación" in line_lower or "alta estimulacion" in line_lower:
                            current_category = "Alta Estimulacion"
                            current_source = "Curaduria_Canales"
                        elif "baja" in line_lower or "moderada" in line_lower:
                            current_category = "Baja/Moderada Estimulacion"
                            current_source = "Curaduria_Canales"
                        continue

                    # Es una URL
                    url = line_str.split(",")[0].strip()
                    label = current_category
                    source = current_source

                    # Soporte para formato CSV opcional URL,label,source en el archivo
                    if "," in line_str:
                        parts = [p.strip() for p in line_str.split(",")]
                        url = parts[0]
                        if len(parts) > 1 and parts[1]:
                            label = parts[1]
                        if len(parts) > 2 and parts[2]:
                            source = parts[2]

                    items_to_process.append((url, label, source))

    # Eliminar duplicados de URLs preservando orden
    seen_urls = set()
    unique_items = []
    for item in items_to_process:
        url = item[0]
        if url not in seen_urls:
            seen_urls.add(url)
            unique_items.append(item)

    if not unique_items:
        print("Error: No se especificaron URLs válidas.")
        sys.exit(1)

    print("=" * 70)
    print(" 🎬 EXTRACTOR DE DATASET DE ESTÍMULO AUDIOVISUAL (YOUTUBE -> CSV)")
    print("=" * 70)
    print(f" • Videos a procesar      : {len(unique_items)}")
    print(f" • Duración por video     : Primeros {args.max_duration} segundos")
    print(f" • Calidad de video       : {args.quality}p")
    print(f" • Dataset consolidado    : {args.output}")
    print(f" • Dataset temporal       : {args.temporal_output}")
    print(f" • Conservar videos .mp4  : {'Sí' if args.keep_videos else 'No (auto-limpieza)'}")
    print("=" * 70)

    pipeline = StimulusPipeline(
        output_csv=args.output,
        temporal_csv=args.temporal_output,
        max_duration=args.max_duration,
        max_height=args.quality,
        cleanup_videos=not args.keep_videos
    )

    df_result = pipeline.run(unique_items)

    if not df_result.empty:
        print("\n" + "=" * 70)
        print(" 🎉 PROCESAMIENTO FINALIZADO EXITOSAMENTE")
        print("=" * 70)
        print(f"Registros generados: {len(df_result)}")
        print("\nPrimeras filas del dataset:")
        print(df_result.head())
    else:
        print("\nNo se generaron registros. Revisa las URLs o la conexión a internet.")


if __name__ == "__main__":
    main()
