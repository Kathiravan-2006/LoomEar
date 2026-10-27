"""
SNR Robustness and background noise stress testing.
"""

from typing import Callable, Dict, List
import numpy as np
from ..audio.augment import add_noise_snr
from .metrics import evaluate_anomaly_detector


def evaluate_snr_robustness(
    test_samples: List[object],
    ambience_clip: np.ndarray,
    detector_model: object,
    feature_extractor_func: Callable,
    snr_levels: List[float] = [20.0, 15.0, 10.0, 5.0, 0.0],
) -> Dict[str, List[float]]:
    """
    Evaluates anomaly detection performance under degrading acoustic environments (decreasing SNR).
    Returns SNR curve dictionaries.
    """
    results = {"snr_db": snr_levels, "roc_auc": [], "partial_auc": [], "f1": []}

    y_true = np.array([1 if s.metadata["condition"] != "healthy" else 0 for s in test_samples])

    for snr in snr_levels:
        corrupted_samples = []
        for s in test_samples:
            noisy_audio = add_noise_snr(s.audio, ambience_clip, snr_db=snr)
            
            # Shallow clone with modified audio
            clone_s = type(s)(
                file_id=s.file_id,
                audio=noisy_audio,
                sample_rate=s.sample_rate,
                duration_sec=s.duration_sec,
                metadata=s.metadata,
            )
            corrupted_samples.append(clone_s)

        X_corr, _ = feature_extractor_func(corrupted_samples)
        scores = detector_model.score_samples(X_corr)
        metrics = evaluate_anomaly_detector(y_true, scores)

        results["roc_auc"].append(metrics["roc_auc"])
        results["partial_auc"].append(metrics["partial_auc_0.10"])
        results["f1"].append(metrics["f1"])

    return results
