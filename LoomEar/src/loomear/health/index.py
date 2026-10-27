"""
Health index calculation, percentile calibration, and temporal smoothing.
Converts raw acoustic anomaly scores into an intuitive, calibrated 0-100 degradation metric.
"""

from typing import Dict, List, Optional, Union
import numpy as np


def compute_health_index(
    anomaly_scores: np.ndarray,
    healthy_baseline_scores: np.ndarray,
    healthy_percentile: float = 90.0,
) -> np.ndarray:
    """
    Transforms raw anomaly scores to a standardized 0 to 100 Health Index:
    - Index < 50: Typical healthy operational range
    - Index = 50: Corresponds to 90th percentile of healthy baseline
    - Index = 60-80: Watch zone (early incipient degradation)
    - Index > 80: Critical Act zone (imminent breakdown risk)
    """
    p_healthy = float(np.percentile(healthy_baseline_scores, healthy_percentile)) + 1e-6
    # Score ratio relative to healthy benchmark
    ratio = anomaly_scores / p_healthy
    
    # Sigmoidal mapping anchored so ratio=1.0 maps to index=50, ratio=2.0 maps to index=80
    # index = 100 / (1 + exp(-k * (ratio - 1.0)))
    k = 1.6
    index = 100.0 / (1.0 + np.exp(-k * (ratio - 1.0)))
    return np.clip(index, 0.0, 100.0)


def smooth_health_index(
    index_series: np.ndarray,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Applies Exponentially Weighted Moving Average (EWMA) to prevent transient acoustic spikes.
    """
    smoothed = np.zeros_like(index_series, dtype=float)
    if len(index_series) == 0:
        return smoothed

    smoothed[0] = index_series[0]
    for t in range(1, len(index_series)):
        smoothed[t] = alpha * index_series[t] + (1.0 - alpha) * smoothed[t - 1]
    return smoothed


class HealthIndexEngine:
    """
    Stateful health index calculator with streaming EWMA and hysteresis buffer.
    """

    def __init__(
        self,
        healthy_baseline_scores: np.ndarray,
        alpha: float = 0.25,
        healthy_percentile: float = 90.0,
    ):
        self.healthy_baseline_scores = healthy_baseline_scores
        self.alpha = alpha
        self.healthy_percentile = healthy_percentile
        self.p_healthy = float(np.percentile(healthy_baseline_scores, healthy_percentile)) + 1e-6
        self.current_smoothed_index = 25.0

    def update(self, raw_anomaly_score: float) -> float:
        """Processes a single cycle anomaly score and returns smoothed health index."""
        raw_idx = compute_health_index(
            np.array([raw_anomaly_score]),
            self.healthy_baseline_scores,
            self.healthy_percentile,
        )[0]
        self.current_smoothed_index = self.alpha * raw_idx + (1.0 - self.alpha) * self.current_smoothed_index
        return float(self.current_smoothed_index)
