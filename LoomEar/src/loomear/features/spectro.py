"""
Spectrogram feature extraction and caching for deep model ingestion.
"""

from typing import List, Tuple
import numpy as np
import librosa


def extract_logmel_dataset(
    samples: List[object],
    n_mels: int = 64,
    fixed_time_bins: int = 64,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extracts fixed 2D log-mel spectrograms of shape (N, 1, n_mels, fixed_time_bins) for all samples.
    """
    specs = []
    for sample in samples:
        audio = sample.audio
        # Compute log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=len(audio) // fixed_time_bins if len(audio) > fixed_time_bins * 1024 else 256,
            n_mels=n_mels,
            fmin=50.0,
            fmax=7800.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Interpolate/crop to exact fixed_time_bins
        if log_mel.shape[1] != fixed_time_bins:
            x_old = np.linspace(0, 1, log_mel.shape[1])
            x_new = np.linspace(0, 1, fixed_time_bins)
            resampled = np.zeros((n_mels, fixed_time_bins), dtype=np.float32)
            for m in range(n_mels):
                resampled[m] = np.interp(x_new, x_old, log_mel[m])
            log_mel = resampled

        # Min-max scale to [0, 1] range
        log_mel_norm = (log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + 1e-6)
        specs.append(log_mel_norm)

    # Shape: (N, 1, H, W)
    dataset = np.expand_dims(np.array(specs, dtype=np.float32), axis=1)
    return dataset
