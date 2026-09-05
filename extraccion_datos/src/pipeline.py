"""
Módulo de orquestación del pipeline completo de scraping, extracción multimodal y exportación a CSV.
"""

import os
import sys
import logging
import pandas as pd
from typing import List, Optional, Any, Union
from tqdm import tqdm
from colorama import Fore, Style, init

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

init(autoreset=True)
logger = logging.getLogger(__name__)

from .downloader import YouTubeDownloader
from .video_analyzer import VideoStimulusAnalyzer
from .audio_analyzer import AudioStimulusAnalyzer


class StimulusPipeline:
    """Orquestador del flujo de procesamiento de estímulo audiovisual."""

    def __init__(
        self,
        output_csv: str = "dataset_estimulacion.csv",
        temporal_csv: str = "dataset_estimulacion_temporal.csv",
        temp_dir: str = "temp_videos",
        max_duration: int = 120,
        max_height: int = 360,
        cleanup_videos: bool = True
    ):
        self.output_csv = output_csv
        self.temporal_csv = temporal_csv
        self.temp_dir = temp_dir
        self.max_duration = max_duration
        self.max_height = max_height
        self.cleanup_videos = cleanup_videos

        self.downloader = YouTubeDownloader(
            output_dir=self.temp_dir,
            max_height=self.max_height,
            max_duration=self.max_duration
        )
        self.video_analyzer = VideoStimulusAnalyzer(target_width=360)
        self.audio_analyzer = AudioStimulusAnalyzer(target_sr=22050)

    def _infer_metadata(self, canal: str, titulo: str) -> dict:
        """Infiere variables de control metodológico (tipo de producción y formato de contenido)."""
        c_lower = (canal or "").lower()
        t_lower = (titulo or "").lower()

        # 1. Tipo de producción: Acción Real vs Animación
        if "blippi" in c_lower or "blippi" in t_lower:
            tipo_prod = "accion_real"
        else:
            tipo_prod = "animacion"

        # 2. Formato: Musical vs Narrativo vs Educativo Exploratorio
        musical_keywords = ["cocomelon", "pinkfong", "looloo", "baby bum", "little baby bum", "song", "cancion", "canciones", "rhymes", "nursery"]
        if any(k in c_lower for k in ["cocomelon", "pinkfong", "looloo", "baby bum"]) or any(k in t_lower for k in ["song", "cancion", "canciones", "rhyme", "nursery"]):
            formato = "musical"
        elif "blippi" in c_lower or "blippi" in t_lower:
            formato = "educativo_exploratorio"
        elif any(k in c_lower for k in ["bluey", "daniel", "pocoyo", "puffin", "peppa"]):
            formato = "narrativo"
        else:
            formato = "narrativo" if tipo_prod == "animacion" else "otro"

        return {"tipo_produccion": tipo_prod, "formato": formato}

    def process_url(self, item, default_label: str = "No especificado", default_source: str = "Sin etiqueta") -> Optional[dict]:
        """Procesa una URL individual (o tupla con label) y devuelve las métricas agregadas y su DataFrame temporal."""
        if isinstance(item, tuple) or isinstance(item, list):
            url = str(item[0]).strip()
            label = str(item[1]).strip() if len(item) > 1 else default_label
            source = str(item[2]).strip() if len(item) > 2 else default_source
        elif isinstance(item, dict):
            url = str(item.get("url", "")).strip()
            label = str(item.get("label", default_label)).strip()
            source = str(item.get("source", default_source)).strip()
        else:
            url = str(item).strip()
            label = default_label
            source = default_source

        if not url or url.startswith("#"):
            return None

        print(f"\n{Fore.CYAN}=== Procesando: {url} [{label}] ==={Style.RESET_ALL}")
        
        # 1. Descarga
        print(f"{Fore.YELLOW}[1/3] Descargando primeros {self.max_duration}s en {self.max_height}p...{Style.RESET_ALL}")
        meta = self.downloader.download_segment(url)
        if not meta:
            print(f"{Fore.RED}[!] Falló la descarga de {url}{Style.RESET_ALL}")
            return None

        filepath = meta["filepath"]
        video_id = meta["video_id"]

        try:
            # 2. Análisis Visual (Cortes, Farneback, Color, Bordes)
            print(f"{Fore.YELLOW}[2/3] Analizando estímulo visual (Cortes, Flujo Óptico, Color, Bordes)...{Style.RESET_ALL}")
            vis_summary, df_vis_temp = self.video_analyzer.analyze(filepath, video_id=video_id)

            # 3. Análisis de Audio (RMS, BPM, Onsets, Brillo Espectral)
            print(f"{Fore.YELLOW}[3/3] Analizando estímulo acústico (RMS, BPM, Onsets, Brillo Espectral)...{Style.RESET_ALL}")
            aud_summary, df_aud_temp = self.audio_analyzer.analyze(filepath, video_id=video_id)

            # Inferir metadatos de control (tipo de producción y formato de contenido)
            meta_inferred = self._infer_metadata(meta.get("canal", ""), meta.get("titulo", ""))

            # Consolidar métricas del video (con etiquetas de Machine Learning)
            video_record = {
                "video_id": meta["video_id"],
                "titulo": meta["titulo"],
                "canal": meta["canal"],
                "categoria_estimulacion": label,
                "tipo_produccion": meta_inferred["tipo_produccion"],
                "formato": meta_inferred["formato"],
                "fuente_etiqueta": source,
                "url": meta["url"],
                "duracion_analizada_seg": vis_summary.get("duracion_analizada_seg", 0.0),
                "cortes_por_minuto": vis_summary.get("cortes_por_minuto", 0.0),
                "duracion_promedio_escena": vis_summary.get("duracion_promedio_escena", 0.0),
                "magnitud_movimiento_promedio": vis_summary.get("magnitud_movimiento_promedio", 0.0),
                "variabilidad_movimiento": vis_summary.get("variabilidad_movimiento", 0.0),
                "luminosidad_promedio": vis_summary.get("luminosidad_promedio", 0.0),
                "variabilidad_luminosidad": vis_summary.get("variabilidad_luminosidad", 0.0),
                "contraste_promedio": vis_summary.get("contraste_promedio", 0.0),
                "variabilidad_contraste": vis_summary.get("variabilidad_contraste", 0.0),
                "saturacion_promedio": vis_summary.get("saturacion_promedio", 0.0),
                "variabilidad_saturacion": vis_summary.get("variabilidad_saturacion", 0.0),
                "flicker_promedio": vis_summary.get("flicker_promedio", 0.0),
                "destellos_por_minuto": vis_summary.get("destellos_por_minuto", 0.0),
                "riqueza_cromatica": vis_summary.get("riqueza_cromatica", 0.0),
                "densidad_bordes_promedio": vis_summary.get("densidad_bordes_promedio", 0.0),
                "energia_audio_promedio": aud_summary.get("energia_audio_promedio", 0.0),
                "variabilidad_audio": aud_summary.get("variabilidad_audio", 0.0),
                "tempo_bpm": aud_summary.get("tempo_bpm", 0.0),
                "onsets_por_minuto": aud_summary.get("onsets_por_minuto", 0.0),
                "brillo_espectral_promedio": aud_summary.get("brillo_espectral_promedio", 0.0)
            }

            # Fusionar series temporales
            if not df_vis_temp.empty and not df_aud_temp.empty:
                df_temp_merged = pd.merge(df_vis_temp, df_aud_temp, on=["video_id", "second"], how="outer").sort_values("second")
            elif not df_vis_temp.empty:
                df_temp_merged = df_vis_temp
            else:
                df_temp_merged = df_aud_temp

            print(f"{Fore.GREEN}[✓] Análisis completado con éxito para '{meta['titulo']}'{Style.RESET_ALL}")
            return {
                "record": video_record,
                "temporal_df": df_temp_merged
            }

        except Exception as e:
            print(f"{Fore.RED}[!] Error procesando el video {url}: {e}{Style.RESET_ALL}")
            logger.exception(f"Error procesando {url}")
            return None

        finally:
            if self.cleanup_videos:
                self.downloader.cleanup_file(filepath)

    def run(self, items: List[Any]) -> pd.DataFrame:
        """Ejecuta el pipeline sobre una lista de URLs o tuplas (url, label, source) y guarda los resultados."""
        if not items:
            print(f"{Fore.YELLOW}No se proporcionaron elementos para procesar.{Style.RESET_ALL}")
            return pd.DataFrame()

        records = []
        temporal_dfs = []

        print(f"\n{Fore.MAGENTA}Iniciando procesamiento de {len(items)} video(s)...{Style.RESET_ALL}")

        for item in tqdm(items, desc="Procesando videos"):
            res = self.process_url(item)
            if res:
                records.append(res["record"])
                if res["temporal_df"] is not None and not res["temporal_df"].empty:
                    temporal_dfs.append(res["temporal_df"])

        if not records:
            print(f"{Fore.RED}No se pudo procesar ningún video correctamente.{Style.RESET_ALL}")
            return pd.DataFrame()

        # Guardar dataset principal
        df_final = pd.DataFrame(records)
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        df_final.to_csv(self.output_csv, index=False, encoding="utf-8-sig")
        print(f"\n{Fore.GREEN}✓ Dataset principal guardado en: {os.path.abspath(self.output_csv)}{Style.RESET_ALL}")

        # Guardar serie temporal secundaria
        if temporal_dfs:
            df_temporal_all = pd.concat(temporal_dfs, ignore_index=True)
            os.makedirs(os.path.dirname(os.path.abspath(self.temporal_csv)), exist_ok=True)
            df_temporal_all.to_csv(self.temporal_csv, index=False, encoding="utf-8-sig")
            print(f"{Fore.GREEN}✓ Serie temporal guardada en: {os.path.abspath(self.temporal_csv)}{Style.RESET_ALL}")

        # Limpiar carpeta temporal si quedó vacía
        if self.cleanup_videos and os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except Exception:
                pass

        return df_final
