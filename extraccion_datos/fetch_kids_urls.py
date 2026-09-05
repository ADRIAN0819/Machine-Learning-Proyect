"""
Script para buscar y extraer automáticamente 100 URLs de videos de YouTube
enfocados en canales infantiles de Alta Estimulación vs Baja/Moderada Estimulación.
"""

import sys
import os
import yt_dlp

# Asegurar codificación UTF-8 en consola de Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Canales de Alta Estimulación (10 videos c/u = 50 videos)
high_stim_sources = [
    ("CoComelon", "https://www.youtube.com/@CoComelon/videos"),
    ("Blippi Español", "https://www.youtube.com/@BlippiEspanol/videos"),
    ("Pinkfong Español", "ytsearch10:Pinkfong en español canciones infantiles"),
    ("LooLoo Kids Español", "https://www.youtube.com/@LooLooKidsEspanol/videos"),
    ("Little Baby Bum Español", "https://www.youtube.com/@LittleBabyBumEspanol/videos")
]

# Canales de Baja / Moderada Estimulación (10 videos c/u = 50 videos)
low_stim_sources = [
    ("Bluey Español", "https://www.youtube.com/@BlueyEs/videos"),
    ("Daniel Tigre Español", "ytsearch10:Daniel Tigre en español episodios"),
    ("Pocoyó", "https://www.youtube.com/@pocoyo/videos"),
    ("Puffin Rock", "ytsearch10:Puffin Rock español episodios completos"),
    ("Peppa Pig Español Latino", "https://www.youtube.com/@PeppaPigEspanolLatinoOficial/videos")
]

all_sources = high_stim_sources + low_stim_sources
output_file = "urls_ejemplo.txt"

ydl_opts = {
    'extract_flat': True,
    'playlistend': 10,
    'quiet': True,
    'no_warnings': True,
    'ignoreerrors': True
}


def fetch_urls():
    print("=" * 70)
    print(" 👶 RECOLECTOR DE URLs DE VIDEOS INFANTILES (ALTA VS BAJA ESTIMULACIÓN)")
    print("=" * 70)
    print(f"Fuentes a consultar: {len(all_sources)} categorías (10 videos c/u = 100 videos)")
    print("=" * 70)

    categorized_urls = []
    total_found = 0

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, (name, source) in enumerate(all_sources, start=1):
            category_type = "Alta Estimulación" if idx <= 5 else "Baja/Moderada Estimulación"
            print(f"[{idx:02d}/{len(all_sources):02d}] Extrayendo de '{name}' ({category_type})...", end=" ", flush=True)

            try:
                info = ydl.extract_info(source, download=False)
                if not info or 'entries' not in info:
                    print("⚠️ Sin resultados.")
                    continue

                channel_urls = []
                for entry in info['entries']:
                    if not entry:
                        continue
                    url = entry.get('url') or entry.get('id')
                    if not url:
                        continue
                    if not url.startswith("http"):
                        url = f"https://www.youtube.com/watch?v={url}"
                    if url not in channel_urls:
                        channel_urls.append(url)
                    if len(channel_urls) >= 10:
                        break

                print(f"✓ {len(channel_urls)} videos obtenidos.")
                categorized_urls.append((name, category_type, channel_urls))
                total_found += len(channel_urls)

            except Exception as e:
                print(f"❌ Error: {e}")

    # Escribir todas las URLs en el archivo urls_ejemplo.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# =====================================================================\n")
        f.write("# DATASET DE 100 VIDEOS INFANTILES: ALTA VS BAJA/MODERADA ESTIMULACIÓN\n")
        f.write("# Generado automáticamente por fetch_kids_urls.py\n")
        f.write("# =====================================================================\n\n")

        for name, cat_type, urls in categorized_urls:
            f.write(f"# --- {name.upper()} ({cat_type}) [{len(urls)} videos] ---\n")
            for u in urls:
                f.write(f"{u}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print(f" 🎉 ¡ÉXITO! Se han guardado {total_found} URLs válidas en '{output_file}'.")
    print("=" * 70)
    print("Ahora puedes ejecutar el pipeline completo con:")
    print("   py extract_stimulus_dataset.py --urls-file urls_ejemplo.txt")
    print("=" * 70)


if __name__ == "__main__":
    fetch_urls()
