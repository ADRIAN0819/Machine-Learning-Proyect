"""
Módulo para la descarga selectiva y ultrarrápida de videos de YouTube usando yt-dlp y ffmpeg.
Descarga en baja resolución (360p) y recorta los primeros N segundos de forma instantánea.
"""

import os
import sys
import shutil
import subprocess
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def ensure_ffmpeg_bin() -> str:
    """Garantiza que exista un ejecutable ffmpeg.exe disponible en una carpeta bin local."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bin_dir = os.path.join(base_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    target_ffmpeg = os.path.join(bin_dir, "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")

    if not os.path.exists(target_ffmpeg):
        try:
            import imageio_ffmpeg
            src_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(src_ffmpeg):
                shutil.copyfile(src_ffmpeg, target_ffmpeg)
        except Exception as e:
            logger.warning(f"No se pudo copiar binario de imageio_ffmpeg: {e}")

    # Agregar bin_dir al inicio de PATH
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    return bin_dir if os.path.exists(target_ffmpeg) else ""


class YouTubeDownloader:
    """Descarga optimizada de fragmentos iniciales de video desde YouTube."""

    def __init__(self, output_dir: str = "temp_videos", max_height: int = 360, max_duration: int = 120):
        self.output_dir = output_dir
        self.max_height = max_height
        self.max_duration = max_duration
        self.bin_dir = ensure_ffmpeg_bin()

        os.makedirs(self.output_dir, exist_ok=True)

    def _trim_video(self, input_path: str, duration_sec: int) -> str:
        """Recorta los primeros duration_sec segundos usando FFmpeg con stream copy ultrarrápido."""
        ffmpeg_exe = os.path.join(self.bin_dir, "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
        if not os.path.exists(ffmpeg_exe):
            return input_path

        dir_name = os.path.dirname(input_path)
        base_name, ext = os.path.splitext(os.path.basename(input_path))
        trimmed_path = os.path.join(dir_name, f"{base_name}_cut{ext}")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", "0",
            "-t", str(duration_sec),
            "-i", input_path,
            "-c", "copy",
            "-loglevel", "error",
            trimmed_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                os.remove(input_path)
                return trimmed_path
        except Exception as e:
            logger.warning(f"No se pudo recortar con stream-copy: {e}. Usando video original.")
        return input_path

    def download_segment(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Descarga el video en resolución <= `max_height` y recorta a `max_duration` segundos.
        
        Retorna un diccionario con metadatos y la ruta local del archivo multimedia, o None si falla.
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("yt-dlp no está instalado. Ejecuta 'pip install yt-dlp'.")

        if self.bin_dir and self.bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = self.bin_dir + os.pathsep + os.environ.get("PATH", "")

        # Localizar Node.js para evaluación de firmas y evitar restricciones
        node_path = shutil.which("node") or (r"C:\Program Files\nodejs\node.exe" if os.path.exists(r"C:\Program Files\nodejs\node.exe") else None)
        format_str = f"bestvideo[height<={self.max_height}]+bestaudio/best[height<={self.max_height}]/best"

        ydl_opts = {
            'format': format_str,
            'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web', 'tv_embedded']
                }
            },
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'ignoreerrors': False,
        }

        if node_path:
            ydl_opts['js_runtimes'] = {'node': {'path': node_path}}

        if self.bin_dir:
            ydl_opts['ffmpeg_location'] = self.bin_dir

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    logger.error(f"No se pudo extraer información de la URL: {url}")
                    return None

                video_id = info.get("id", "unknown_id")
                title = info.get("title", "Sin título")
                channel = info.get("uploader") or info.get("channel", "Desconocido")
                duration = info.get("duration", 0)

                # Buscar el archivo generado (puede ser .mp4, .mkv, .webm, etc.)
                video_path = None
                for ext in [".mp4", ".mkv", ".webm", ".avi"]:
                    test_p = os.path.join(self.output_dir, f"{video_id}{ext}")
                    if os.path.exists(test_p):
                        video_path = test_p
                        break

                if not video_path:
                    for f in os.listdir(self.output_dir):
                        if f.startswith(video_id) and not f.endswith(".part"):
                            video_path = os.path.join(self.output_dir, f)
                            break

                if not video_path or not os.path.exists(video_path):
                    logger.error(f"El archivo descargado no se encontró para {video_id}")
                    return None

                # Si el video dura más que max_duration, recortarlo instantáneamente con FFmpeg
                if duration > self.max_duration:
                    video_path = self._trim_video(video_path, self.max_duration)

                return {
                    "video_id": video_id,
                    "titulo": title,
                    "canal": channel,
                    "url": url,
                    "duracion_total_video": duration,
                    "filepath": os.path.abspath(video_path)
                }

        except Exception as e:
            logger.error(f"Error al descargar {url}: {e}")
            return None

    def cleanup_file(self, filepath: str) -> None:
        """Elimina el archivo de video local descargado para ahorrar espacio."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Archivo eliminado: {filepath}")
        except Exception as e:
            logger.warning(f"No se pudo eliminar el archivo {filepath}: {e}")
