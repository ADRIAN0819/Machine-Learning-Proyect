"""
Módulo para el análisis cuantitativo de estímulo visual en videos usando OpenCV y PySceneDetect.
Extrae cortes/ritmo, flujo óptico (Farneback), color, contraste, saturación y densidad de bordes (Canny).
"""

import cv2
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional
from scenedetect import detect, ContentDetector, AdaptiveDetector

logger = logging.getLogger(__name__)


class VideoStimulusAnalyzer:
    """Analizador de variables visuales de estímulo en video."""

    def __init__(self, target_width: int = 360, sample_fps: Optional[float] = None):
        """
        target_width: Ancho al cual redimensionar frames para cómputo de flujo óptico rápido.
        sample_fps: Tasa de muestreo de frames opcional (None = analizar cada frame).
        """
        self.target_width = target_width
        self.sample_fps = sample_fps

    def detect_scenes(self, video_path: str, duration_sec: float) -> Tuple[float, float, int, List[float]]:
        """
        Detecta cortes y tomas usando PySceneDetect.
        Retorna (cortes_por_minuto, duracion_promedio_escena, cortes_totales, lista_tiempos_corte).
        """
        try:
            scene_list = detect(video_path, ContentDetector(threshold=27.0))
            num_scenes = max(1, len(scene_list))
            num_cuts = max(0, len(scene_list) - 1)

            cut_timestamps = []
            for scene in scene_list[:-1]:
                cut_timestamps.append(scene[1].get_seconds())

            eff_duration = max(duration_sec, 1.0)
            cortes_por_minuto = (num_cuts / eff_duration) * 60.0
            
            if scene_list:
                duraciones = [(s[1] - s[0]).get_seconds() for s in scene_list]
                duracion_promedio_escena = float(np.mean(duraciones))
            else:
                duracion_promedio_escena = eff_duration

            return cortes_por_minuto, duracion_promedio_escena, num_cuts, cut_timestamps
        except Exception as e:
            logger.warning(f"Error en PySceneDetect para {video_path}: {e}. Usando estimación fallback.")
            return 0.0, max(duration_sec, 1.0), 0, []

    def analyze(self, video_path: str, video_id: str = "") -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Procesa el archivo .mp4 fotograma a fotograma.
        
        Retorna:
            - summary_dict: Métricas cuantitativas agregadas del video.
            - temporal_df: Serie temporal segundo a segundo de las métricas visuales.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        # Análisis de cortes con PySceneDetect
        cortes_por_minuto, duracion_promedio_escena, total_cortes, cut_timestamps = self.detect_scenes(video_path, duration_sec)

        # Preparar escala de redimensionamiento si target_width está definido
        resize_dim = None
        if self.target_width and orig_width > self.target_width:
            aspect_ratio = orig_height / orig_width
            resize_dim = (self.target_width, int(self.target_width * aspect_ratio))

        # Almacenamiento frame a frame
        frame_metrics = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_time_sec = frame_idx / fps

            if resize_dim:
                frame_resized = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame

            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]

            # 1. Luminosidad y Contraste
            lum_val = float(np.mean(v_channel))
            cont_val = float(np.std(v_channel))

            # 2. Saturación de color
            sat_val = float(np.mean(s_channel))

            # 3. Riqueza cromática (Entropía y varianza del canal Hue)
            hist, _ = np.histogram(h_channel, bins=36, range=(0, 180), density=True)
            hist_nonzero = hist[hist > 0]
            hue_entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
            # Escalamos la entropía a rango orientativo [0 - 100] (máx teórico log2(36) ≈ 5.17)
            riqueza_cromatica = float(min(100.0, (hue_entropy / 5.17) * 100.0))

            # 4. Densidad de bordes (Canny)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float((np.count_nonzero(edges) / edges.size) * 100.0)

            # 5. Flujo Óptico (Farneback)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=2, winsize=13,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mag = float(np.mean(mag))
            else:
                motion_mag = 0.0

            prev_gray = gray

            frame_metrics.append({
                "second": int(current_time_sec),
                "timestamp": current_time_sec,
                "motion": motion_mag,
                "luminosity": lum_val,
                "contrast": cont_val,
                "saturation": sat_val,
                "chroma_richness": riqueza_cromatica,
                "edge_density": edge_density
            })

            frame_idx += 1

        cap.release()

        if not frame_metrics:
            logger.warning(f"No se pudieron leer frames de {video_path}")
            return {}, pd.DataFrame()

        df_frames = pd.DataFrame(frame_metrics)
        # Ignorar primer frame para movimiento (es 0.0 por inicialización)
        motion_series = df_frames["motion"].iloc[1:] if len(df_frames) > 1 else df_frames["motion"]

        # 1. Variabilidad temporal inter-frame de luz y color (std a través del tiempo)
        lum_arr = df_frames["luminosity"].to_numpy()
        cont_arr = df_frames["contrast"].to_numpy()
        sat_arr = df_frames["saturation"].to_numpy()

        var_luminosidad = float(np.std(lum_arr))
        var_contraste = float(np.std(cont_arr))
        var_saturacion = float(np.std(sat_arr))

        # 2. Flicker (diferencia de luminosidad frame a frame) y conteo de destellos bruscos
        if len(lum_arr) > 1:
            lum_diff = np.abs(np.diff(lum_arr))
            flicker_promedio = float(np.mean(lum_diff))
            # Destellos bruscos: saltos de luminosidad >= 15 puntos (en escala 0-255)
            flashes_count = int(np.count_nonzero(lum_diff >= 15.0))
            destellos_por_minuto = float((flashes_count / max(duration_sec, 1.0)) * 60.0)
            df_frames["flicker"] = np.insert(lum_diff, 0, 0.0)
        else:
            flicker_promedio = 0.0
            destellos_por_minuto = 0.0
            df_frames["flicker"] = 0.0

        # Agregados globales de video
        summary = {
            "duracion_analizada_seg": round(duration_sec, 2),
            "cortes_por_minuto": round(cortes_por_minuto, 2),
            "duracion_promedio_escena": round(duracion_promedio_escena, 2),
            "cortes_totales": total_cortes,
            "magnitud_movimiento_promedio": round(float(motion_series.mean()), 4),
            "variabilidad_movimiento": round(float(motion_series.std(ddof=0)), 4),
            "luminosidad_promedio": round(float(np.mean(lum_arr)), 2),
            "variabilidad_luminosidad": round(var_luminosidad, 2),
            "contraste_promedio": round(float(np.mean(cont_arr)), 2),
            "variabilidad_contraste": round(var_contraste, 2),
            "saturacion_promedio": round(float(np.mean(sat_arr)), 2),
            "variabilidad_saturacion": round(var_saturacion, 2),
            "flicker_promedio": round(flicker_promedio, 4),
            "destellos_por_minuto": round(destellos_por_minuto, 2),
            "riqueza_cromatica": round(float(df_frames["chroma_richness"].mean()), 2),
            "densidad_bordes_promedio": round(float(df_frames["edge_density"].mean()), 2),
        }

        # Agregación temporal segundo a segundo
        df_temporal = df_frames.groupby("second").agg({
            "motion": "mean",
            "luminosity": "mean",
            "contrast": "mean",
            "saturation": "mean",
            "flicker": "mean",
            "chroma_richness": "mean",
            "edge_density": "mean"
        }).reset_index()

        # Marcar cortes por segundo
        cut_counts = {}
        for c_t in cut_timestamps:
            sec = int(c_t)
            cut_counts[sec] = cut_counts.get(sec, 0) + 1

        df_temporal["cortes_segundo"] = df_temporal["second"].map(cut_counts).fillna(0).astype(int)
        df_temporal.insert(0, "video_id", video_id)

        return summary, df_temporal
