"""
Audio preprocessing and conditioning for power loom acoustics.
Includes DC removal, bandpass filtering, peak normalization, and stationary noise suppression.
"""

from typing import Optional
import logging
import numpy as np
from scipy import signal
import noisereduce as nr

logger = logging.getLogger(__name__)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Removes DC offset by subtracting mean."""
    return audio - np.mean(audio)


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int = 16000,
    low_hz: float = 50.0,
    high_hz: float = 7800.0,
    order: int = 4,
) -> np.ndarray:
    """
    Applies Butterworth zero-phase bandpass filter to eliminate sub-audible mechanical rumble
    and ultrasonic interference above Nyquist guard band.
    """
    nyquist = 0.5 * sample_rate
    low = max(1e-3, low_hz / nyquist)
    high = min(0.999, high_hz / nyquist)
    
    if low >= high:
        raise ValueError(f"Invalid filter frequencies: low={low_hz}Hz >= high={high_hz}Hz")

    b, a = signal.butter(order, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalizes the audio to target peak value."""
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return (audio / peak * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = 16000,
    noise_clip: Optional[np.ndarray] = None,
    prop_decrease: float = 0.85,
    stationary: bool = True,
) -> np.ndarray:
    """
    Suppresses continuous stationary background factory noise using spectral gating.
    """
    try:
        if noise_clip is not None and len(noise_clip) > 0:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=stationary,
            )
        else:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=prop_decrease,
                stationary=stationary,
            )
        return cleaned.astype(np.float32)
    except Exception as e:
        logger.warning(f"Noise reduction failed ({str(e)}), returning unreduced audio.")
        return audio.astype(np.float32)


def clean_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    low_hz: float = 50.0,
    high_hz: float = 7800.0,
    enable_noise_reduction: bool = False,
    noise_clip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full preprocessing pipeline for incoming loom audio stream/file."""
    dc_free = remove_dc(audio)
    filtered = bandpass_filter(dc_free, sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz)
    
    if enable_noise_reduction:
        filtered = reduce_noise(filtered, sample_rate=sample_rate, noise_clip=noise_clip)
        
    normalized = normalize_audio(filtered)
    return normalized
