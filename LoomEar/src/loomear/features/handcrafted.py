"""
Handcrafted acoustic feature extraction for power loom fault diagnostics.
Extracts time-domain, frequency-domain, perceptual (MFCC), and cycle-synchronous features.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal, stats
import librosa

FEATURE_NAMES = [
    # Time Domain (8)
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "zcr",
    "impulse_factor",
    "margin_factor",
    # Frequency Domain (9)
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_flux",
    "band_energy_low",
    "band_energy_mid",
    "band_energy_high",
    "high_to_low_energy_ratio",
    # Perceptual (26: 13 MFCC + 13 Delta)
    *[f"mfcc_{i}" for i in range(13)],
    *[f"mfcc_delta_{i}" for i in range(13)],
    # Cycle Synchronous (7)
    "impact_peak_offset",
    "n_peaks_per_cycle",
    "cycle_rms_variance",
    "residual_energy_early",
    "residual_energy_mid",
    "residual_energy_late",
    "template_correlation",
]


def get_feature_names() -> List[str]:
    """Returns the list of all extracted feature names."""
    return list(FEATURE_NAMES)


def extract_time_features(audio: np.ndarray) -> Dict[str, float]:
    """Extracts statistical time-domain features sensitive to impacts and friction."""
    eps = 1e-12
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(audio ** 2))) + eps
    peak = float(np.max(abs_audio))
    crest_factor = float(peak / rms)
    kurt = float(stats.kurtosis(audio))
    skew = float(stats.skew(audio))
    
    # Zero crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    
    mean_abs = float(np.mean(abs_audio)) + eps
    impulse_factor = float(peak / mean_abs)
    
    square_root_amp = np.mean(np.sqrt(abs_audio)) ** 2 + eps
    margin_factor = float(peak / square_root_amp)

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "skewness": skew,
        "zcr": zcr,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor,
    }


def extract_frequency_features(audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
    """Extracts spectral shape, energy band distributions, and spectral flux."""
    eps = 1e-12
    
    # STFT & Spectrogram
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)

    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sample_rate)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sample_rate, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    # Spectral flux
    diff = np.diff(np.sqrt(S), axis=1)
    flux = float(np.mean(np.sum(np.maximum(0, diff) ** 2, axis=0)))

    # Band energy ratios
    # Low: 50 - 500 Hz (Beat-up, frame vibration)
    # Mid: 500 - 3000 Hz (Shedding, rapier motion, belt squeal)
    # High: 3000 - 7800 Hz (Bearing wear, metal-to-metal dry friction)
    low_mask = (freqs >= 50) & (freqs < 500)
    mid_mask = (freqs >= 500) & (freqs < 3000)
    high_mask = (freqs >= 3000) & (freqs <= 7800)

    total_energy = np.sum(S) + eps
    energy_low = float(np.sum(S[low_mask, :]) / total_energy)
    energy_mid = float(np.sum(S[mid_mask, :]) / total_energy)
    energy_high = float(np.sum(S[high_mask, :]) / total_energy)
    high_to_low = float((energy_high + eps) / (energy_low + eps))

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
        "band_energy_low": energy_low,
        "band_energy_mid": energy_mid,
        "band_energy_high": energy_high,
        "high_to_low_energy_ratio": high_to_low,
    }


def extract_perceptual_features(audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
    """Extracts 13 MFCC means and 13 Delta-MFCC means."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_deltas = librosa.feature.delta(mfccs)

    feats = {}
    for i in range(13):
        feats[f"mfcc_{i}"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_delta_{i}"] = float(np.mean(mfcc_deltas[i]))
    return feats


def extract_cycle_synchronous_features(
    audio: np.ndarray,
    normalized_cycles: np.ndarray,
    healthy_template: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Extracts features tied directly to the loom pick cycle structure:
    - Timing position of the main impact peak relative to pick period
    - Cycle-to-cycle amplitude stability
    - Energy residual in early (shedding), mid (beat-up), and late (weft cut) pick phases
    """
    if len(normalized_cycles) == 0:
        normalized_cycles = np.zeros((1, 4000), dtype=np.float32)

    mean_cycle = np.mean(normalized_cycles, axis=0)
    cycle_len = len(mean_cycle)

    # 1. Main impact peak offset (0.0 to 1.0 of cycle)
    peak_idx = int(np.argmax(np.abs(mean_cycle)))
    impact_offset = float(peak_idx / max(1, cycle_len))

    # 2. Number of prominent peaks per cycle (detects doubled impacts / sley looseness)
    peaks, _ = signal.find_peaks(np.abs(mean_cycle), height=0.3 * np.max(np.abs(mean_cycle)) + 1e-6)
    n_peaks = float(len(peaks))

    # 3. Cycle-to-cycle RMS variance
    cycle_rms = np.sqrt(np.mean(normalized_cycles ** 2, axis=1) + 1e-12)
    cycle_rms_var = float(np.std(cycle_rms))

    # 4. Phase-segmented residual energy
    # Early (0 - 30%), Mid (30% - 60%), Late (60% - 100%)
    p1 = int(0.30 * cycle_len)
    p2 = int(0.60 * cycle_len)

    if healthy_template is not None and "mean_waveform" in healthy_template:
        ref_wave = healthy_template["mean_waveform"]
        diff = mean_cycle - ref_wave[:cycle_len]
        res_early = float(np.sum(diff[:p1] ** 2))
        res_mid = float(np.sum(diff[p1:p2] ** 2))
        res_late = float(np.sum(diff[p2:] ** 2))
        
        # Pearson correlation with healthy template
        corr, _ = stats.pearsonr(mean_cycle, ref_wave[:cycle_len])
        template_corr = float(corr) if not np.isnan(corr) else 1.0
    else:
        res_early = float(np.sum(mean_cycle[:p1] ** 2))
        res_mid = float(np.sum(mean_cycle[p1:p2] ** 2))
        res_late = float(np.sum(mean_cycle[p2:] ** 2))
        template_corr = 1.0

    return {
        "impact_peak_offset": impact_offset,
        "n_peaks_per_cycle": n_peaks,
        "cycle_rms_variance": cycle_rms_var,
        "residual_energy_early": res_early,
        "residual_energy_mid": res_mid,
        "residual_energy_late": res_late,
        "template_correlation": template_corr,
    }


def extract_features_for_sample(
    audio: np.ndarray,
    sample_rate: int = 16000,
    normalized_cycles: Optional[np.ndarray] = None,
    healthy_template: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Extracts all 50 features for a single audio recording."""
    if normalized_cycles is None:
        from ..audio.cycles import segment_cycles
        res = segment_cycles(audio, sample_rate=sample_rate)
        normalized_cycles = res.normalized_cycles

    feats = {}
    feats.update(extract_time_features(audio))
    feats.update(extract_frequency_features(audio, sample_rate=sample_rate))
    feats.update(extract_perceptual_features(audio, sample_rate=sample_rate))
    feats.update(extract_cycle_synchronous_features(audio, normalized_cycles, healthy_template))
    return feats


def extract_handcrafted_features(
    samples: List[object],
    templates_by_machine: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts complete feature matrix X for a list of AudioSample objects.
    """
    feature_names = get_feature_names()
    rows = []

    for sample in samples:
        audio = sample.audio
        sr = sample.sample_rate
        machine_id = str(sample.metadata.get("machine_id", "default"))
        template = templates_by_machine.get(machine_id) if templates_by_machine else None

        f_dict = extract_features_for_sample(audio, sample_rate=sr, healthy_template=template)
        row = [f_dict.get(name, 0.0) for name in feature_names]
        rows.append(row)

    X = np.array(rows, dtype=np.float32)
    # NaN / Inf safety guard
    X = np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)
    return X, feature_names
