"""
Data augmentation for power loom audio and spectrograms.
Includes SpecAugment (frequency & time masking) and SNR-controlled background noise injection.
"""

from typing import Optional
import numpy as np


def spec_augment(
    spectrogram: np.ndarray,
    max_time_mask: int = 8,
    max_freq_mask: int = 8,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Applies SpecAugment frequency and time masking to 2D log-mel spectrogram.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = spectrogram.copy()
    n_mels, n_time = augmented.shape
    fill_val = float(np.mean(augmented))

    # Frequency masking
    for _ in range(num_freq_masks):
        f_len = rng.integers(1, min(max_freq_mask, n_mels))
        f_start = rng.integers(0, n_mels - f_len)
        augmented[f_start : f_start + f_len, :] = fill_val

    # Time masking
    for _ in range(num_time_masks):
        t_len = rng.integers(1, min(max_time_mask, n_time))
        t_start = rng.integers(0, n_time - t_len)
        augmented[:, t_start : t_start + t_len] = fill_val

    return augmented


def add_noise_snr(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 15.0,
) -> np.ndarray:
    """
    Mixes background noise into audio at a specified SNR (dB).
    """
    n = len(audio)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))[:n]
    else:
        noise = noise[:n]

    p_sig = np.mean(audio ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12

    target_p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_p_noise / p_noise)

    mixed = audio + scale * noise
    return mixed.astype(np.float32)
