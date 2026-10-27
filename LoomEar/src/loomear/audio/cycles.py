"""
Pick-cycle-synchronous audio analysis for power looms.
Transforms asynchronous acoustic streams into cycle-aligned, phase-normalized templates and residuals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class CycleExtractionResult:
    """Output container for cycle segmentation and period estimation."""
    estimated_rpm: float
    estimated_period_sec: float
    confidence: float
    cycle_boundaries: List[int]
    cycles: List[np.ndarray]
    normalized_cycles: np.ndarray  # Shape: (N_cycles, target_samples)


def compute_envelope(
    audio: np.ndarray,
    sample_rate: int = 16000,
    cutoff_hz: float = 30.0,
) -> np.ndarray:
    """
    Computes smoothed signal envelope using Hilbert analytic amplitude
    followed by low-pass smoothing to isolate macro-cycle dynamics.
    """
    analytic_signal = signal.hilbert(audio)
    raw_envelope = np.abs(analytic_signal)

    # Low-pass filter to smooth micro-transients
    nyquist = 0.5 * sample_rate
    b, a = signal.butter(3, min(0.99, cutoff_hz / nyquist), btype="low")
    smoothed = signal.filtfilt(b, a, raw_envelope)
    return np.maximum(0.0, smoothed).astype(np.float32)


def estimate_pick_period(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_rpm: float = 120.0,
    max_rpm: float = 400.0,
    envelope_cutoff_hz: float = 30.0,
) -> Tuple[float, float, float]:
    """
    Estimates the exact loom pick period using autocorrelation of the envelope.
    Returns: (estimated_rpm, period_sec, confidence_score)
    """
    envelope = compute_envelope(audio, sample_rate, cutoff_hz=envelope_cutoff_hz)
    env_norm = envelope - np.mean(envelope)
    
    # Fast FFT-based Autocorrelation
    n = len(env_norm)
    fft_res = np.fft.rfft(env_norm, n=2 * n)
    autocorr = np.fft.irfft(fft_res * np.conj(fft_res))[:n]
    if autocorr[0] > 1e-12:
        autocorr /= autocorr[0]

    # Convert RPM bounds to sample lags
    # period_sec = 60 / rpm  => lag_samples = period_sec * sample_rate
    min_lag = int(round((60.0 / max_rpm) * sample_rate))
    max_lag = int(round((60.0 / min_rpm) * sample_rate))
    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        return min_rpm, 60.0 / min_rpm, 0.0

    search_region = autocorr[min_lag:max_lag]
    peak_rel_idx = np.argmax(search_region)
    best_lag = min_lag + peak_rel_idx

    # Parabolic interpolation for sub-sample precision
    if 0 < peak_rel_idx < len(search_region) - 1:
        y0, y1, y2 = search_region[peak_rel_idx - 1], search_region[peak_rel_idx], search_region[peak_rel_idx + 1]
        denom = 2.0 * (2 * y1 - y0 - y2) + 1e-12
        sub_offset = (y0 - y2) / denom
        refined_lag = best_lag + sub_offset
    else:
        refined_lag = float(best_lag)

    period_sec = refined_lag / sample_rate
    estimated_rpm = 60.0 / period_sec
    confidence = float(np.clip(autocorr[best_lag], 0.0, 1.0))

    return estimated_rpm, period_sec, confidence


def time_normalize_cycle(cycle: np.ndarray, target_samples: int = 4000) -> np.ndarray:
    """
    Interpolates a single cycle of arbitrary duration to exactly target_samples,
    enabling direct cycle-phase alignment.
    """
    if len(cycle) == target_samples:
        return cycle.astype(np.float32)
    if len(cycle) < 4:
        return np.zeros(target_samples, dtype=np.float32)

    orig_indices = np.linspace(0, 1, len(cycle))
    target_indices = np.linspace(0, 1, target_samples)
    normalized = np.interp(target_indices, orig_indices, cycle)
    return normalized.astype(np.float32)


def segment_cycles(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_samples: int = 4000,
    min_rpm: float = 120.0,
    max_rpm: float = 400.0,
) -> CycleExtractionResult:
    """
    Segments an audio recording into individual pick cycles anchored on recurring beat-up events.
    """
    rpm, period_sec, confidence = estimate_pick_period(
        audio, sample_rate, min_rpm=min_rpm, max_rpm=max_rpm
    )
    period_samples = int(round(period_sec * sample_rate))
    envelope = compute_envelope(audio, sample_rate)

    # Find dominant recurring impact peaks (beat-up)
    min_distance = int(0.75 * period_samples)
    peaks, _ = signal.find_peaks(
        envelope,
        distance=min_distance,
        height=np.mean(envelope) + 0.3 * np.std(envelope),
    )

    cycles = []
    normalized_list = []
    cycle_boundaries = []

    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            seg = audio[start:end]
            if 0.5 * period_samples <= len(seg) <= 1.5 * period_samples:
                cycle_boundaries.append(start)
                cycles.append(seg)
                normalized_list.append(time_normalize_cycle(seg, target_samples))
    else:
        # Fallback to periodic grid slicing if peak prominence is low
        for start in range(0, len(audio) - period_samples, period_samples):
            seg = audio[start : start + period_samples]
            cycle_boundaries.append(start)
            cycles.append(seg)
            normalized_list.append(time_normalize_cycle(seg, target_samples))

    if not normalized_list:
        normalized_array = np.zeros((1, target_samples), dtype=np.float32)
    else:
        normalized_array = np.array(normalized_list, dtype=np.float32)

    return CycleExtractionResult(
        estimated_rpm=rpm,
        estimated_period_sec=period_sec,
        confidence=confidence,
        cycle_boundaries=cycle_boundaries,
        cycles=cycles,
        normalized_cycles=normalized_array,
    )


def compute_log_mel_spectrogram(
    audio_segment: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mels: int = 64,
    f_min: float = 50.0,
    f_max: float = 7800.0,
) -> np.ndarray:
    """
    Computes a fixed 2D log-mel spectrogram for a cycle or audio segment.
    """
    # STFT via scipy
    f, t_spec, Zxx = signal.stft(
        audio_segment,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
    )
    power_spec = np.abs(Zxx) ** 2

    # Mel filterbank construction
    nyquist = sample_rate / 2.0
    f_max = min(f_max, nyquist)
    
    # Mel scaling: m = 2595 * log10(1 + f / 700)
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    f_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    
    bin_points = np.floor((n_fft + 1) * f_points / sample_rate).astype(int)
    
    fb = np.zeros((n_mels, len(f)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center > f_left:
                fb[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fb[m - 1, k] = (f_right - k) / (f_right - f_center)

    mel_spec = np.dot(fb, power_spec)
    log_mel = np.log10(np.maximum(mel_spec, 1e-6))
    return log_mel.astype(np.float32)


def build_healthy_template(
    healthy_normalized_cycles: np.ndarray,
    sample_rate: int = 16000,
) -> Dict[str, np.ndarray]:
    """
    Constructs a machine-specific healthy baseline template across many cycles:
    - Mean and standard deviation waveform envelope
    - Median log-mel spectrogram and per-bin standard deviation
    """
    if len(healthy_normalized_cycles) == 0:
        raise ValueError("Cannot build template from empty cycle array.")

    # Compute spectrogram for each cycle
    specs = []
    for cycle in healthy_normalized_cycles:
        spec = compute_log_mel_spectrogram(cycle, sample_rate=sample_rate)
        specs.append(spec)

    specs_arr = np.array(specs)  # (N_cycles, n_mels, time_bins)

    template_mean_spec = np.median(specs_arr, axis=0)
    template_std_spec = np.std(specs_arr, axis=0) + 1e-4

    mean_waveform = np.mean(healthy_normalized_cycles, axis=0)
    std_waveform = np.std(healthy_normalized_cycles, axis=0) + 1e-4

    return {
        "mean_spec": template_mean_spec,
        "std_spec": template_std_spec,
        "mean_waveform": mean_waveform,
        "std_waveform": std_waveform,
    }


def compute_cycle_residual(
    cycle_audio: np.ndarray,
    template: Dict[str, np.ndarray],
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, float]:
    """
    Computes the standardized residual heatmap between a live cycle and the healthy template:
    Residual(f, t) = (S_live(f, t) - S_template(f, t)) / (Sigma_template(f, t) + eps)
    Returns: (residual_matrix, anomaly_score)
    """
    live_spec = compute_log_mel_spectrogram(cycle_audio, sample_rate=sample_rate)
    mean_spec = template["mean_spec"]
    std_spec = template["std_spec"]

    # Align shapes if slight mismatch
    min_time = min(live_spec.shape[1], mean_spec.shape[1])
    live_spec = live_spec[:, :min_time]
    mean_spec = mean_spec[:, :min_time]
    std_spec = std_spec[:, :min_time]

    residual = (live_spec - mean_spec) / std_spec
    # Anomaly score is mean squared standardized residual
    anomaly_score = float(np.mean(residual ** 2))
    return residual.astype(np.float32), anomaly_score


def evaluate_cycle_alignment_gain(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_samples: int = 4000,
) -> Dict[str, float]:
    """
    Quantifies the scientific advantage of pick-cycle alignment versus naive fixed-window chopping:
    Measures the intra-class variance of aligned cycles vs unaligned windows.
    """
    res = segment_cycles(audio, sample_rate=sample_rate, target_samples=target_samples)
    aligned_cycles = res.normalized_cycles
    
    # Naive unaligned fixed windows
    n_windows = len(aligned_cycles)
    unaligned_windows = []
    step = target_samples
    for i in range(0, min(len(audio) - target_samples, n_windows * step), step):
        unaligned_windows.append(audio[i : i + target_samples])
    unaligned_windows = np.array(unaligned_windows)

    aligned_var = float(np.mean(np.var(aligned_cycles, axis=0)))
    unaligned_var = float(np.mean(np.var(unaligned_windows, axis=0))) if len(unaligned_windows) > 1 else aligned_var * 2.0
    
    variance_reduction_pct = max(0.0, (unaligned_var - aligned_var) / (unaligned_var + 1e-12) * 100.0)

    return {
        "aligned_variance": aligned_var,
        "unaligned_variance": unaligned_var,
        "variance_reduction_pct": variance_reduction_pct,
        "estimated_rpm": res.estimated_rpm,
        "period_confidence": res.confidence,
    }
