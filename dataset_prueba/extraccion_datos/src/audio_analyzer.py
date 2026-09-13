"""
Módulo optimizado para el análisis cuantitativo de audio usando FFmpeg + Librosa/SoundFile.
Extrae energía RMS, variabilidad de volumen, tempo estimado (BPM) y brillo espectral (Centroide).
"""

import os
import sys
import subprocess
import shutil
import logging
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def get_ffmpeg_binary() -> Optional[str]:
    """Obtiene la ruta absoluta al ejecutable de FFmpeg."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bin_ffmpeg = os.path.join(base_dir, "bin", "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
    if os.path.exists(bin_ffmpeg):
        return bin_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class AudioStimulusAnalyzer:
    """Analizador de variables acústicas de estímulo usando Librosa y SoundFile."""

    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        self.ffmpeg_path = get_ffmpeg_binary()

    def _extract_pcm_wav(self, media_path: str) -> Optional[str]:
        """Extrae la pista de audio a un archivo WAV PCM ultrarrápido usando FFmpeg."""
        try:
            if not self.ffmpeg_path or not os.path.exists(self.ffmpeg_path):
                self.ffmpeg_path = get_ffmpeg_binary()

            if not self.ffmpeg_path:
                return None

            wav_path = os.path.splitext(media_path)[0] + "_temp_audio.wav"
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", media_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.target_sr),
                "-ac", "1",
                "-loglevel", "error",
                wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return wav_path if os.path.exists(wav_path) else None
        except Exception as e:
            logger.warning(f"Extracción WAV falló para {media_path}: {e}")
            return None

    def analyze(self, media_path: str, video_id: str = "") -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Analiza el audio del video.
        
        Retorna:
            - summary_dict: Métricas cuantitativas agregadas de audio.
            - temporal_df: Serie temporal segundo a segundo del audio.
        """
        wav_path = self._extract_pcm_wav(media_path)
        y = None
        sr = self.target_sr

        if wav_path and os.path.exists(wav_path):
            try:
                y, sr = sf.read(wav_path, dtype='float32')
            except Exception as e:
                logger.warning(f"Error al leer WAV extraído: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        # Fallback si no se pudo extraer WAV
        if y is None or len(y) == 0:
            try:
                y, sr = librosa.load(media_path, sr=self.target_sr, mono=True)
            except Exception as e:
                logger.warning(f"No se pudo cargar audio de {media_path}: {e}")
                y = None

        if y is None or len(y) == 0:
            summary = {
                "energia_audio_promedio": 0.0,
                "variabilidad_audio": 0.0,
                "tempo_bpm": 0.0,
                "brillo_espectral_promedio": 0.0
            }
            return summary, pd.DataFrame(columns=["video_id", "second", "audio_energy", "spectral_brightness"])

        hop_length = 512
        frame_length = 2048

        # 1. Energía de audio (RMS) y su variabilidad
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        energia_audio_promedio = float(np.mean(rms_frames))
        variabilidad_audio = float(np.std(rms_frames))

        # 2. Brillo Espectral (Spectral Centroid)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        brillo_espectral_promedio = float(np.mean(spectral_centroid))

        # 3. Tempo (BPM) y Onsets (Golpes / Ataques por minuto)
        onset_times = []
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            if isinstance(tempo, (np.ndarray, list)):
                tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo_val = float(tempo)

            # Detección de Onsets calibrada (ataques percusivos / golpes de sonido significativos)
            wait_frames = max(1, int(0.12 * sr / hop_length)) # Mínimo 120ms entre ataques distintos
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                delta=0.08,
                wait=wait_frames
            )
            audio_duration_sec = len(y) / sr if sr > 0 else 1.0
            onsets_por_minuto = float((len(onset_frames) / max(audio_duration_sec, 1.0)) * 60.0)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        except Exception as e:
            logger.warning(f"Error al estimar tempo u onsets: {e}")
            tempo_val = 0.0
            onsets_por_minuto = 0.0

        # 4. Serie temporal segundo a segundo
        frame_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)
        df_audio_frames = pd.DataFrame({
            "second": np.floor(frame_times).astype(int),
            "audio_energy": rms_frames,
            "spectral_brightness": spectral_centroid
        })

        df_temporal_audio = df_audio_frames.groupby("second").agg({
            "audio_energy": "mean",
            "spectral_brightness": "mean"
        }).reset_index()

        # Mapear conteo de onsets por segundo
        onset_counts = {}
        for ot in onset_times:
            sec = int(ot)
            onset_counts[sec] = onset_counts.get(sec, 0) + 1
        df_temporal_audio["onsets_segundo"] = df_temporal_audio["second"].map(onset_counts).fillna(0).astype(int)

        df_temporal_audio.insert(0, "video_id", video_id)

        summary = {
            "energia_audio_promedio": round(energia_audio_promedio, 4),
            "variabilidad_audio": round(variabilidad_audio, 4),
            "tempo_bpm": round(tempo_val, 2),
            "onsets_por_minuto": round(onsets_por_minuto, 2),
            "brillo_espectral_promedio": round(brillo_espectral_promedio, 2)
        }

        return summary, df_temporal_audio
