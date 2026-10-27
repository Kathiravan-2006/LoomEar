"""Evaluation metrics, robustness testing, and explainability modules."""

from .metrics import (
    evaluate_anomaly_detector,
    evaluate_classifier,
    compute_partial_auc,
    bootstrap_metric_ci,
)
from .robustness import evaluate_snr_robustness
from .explain import generate_residual_explanation, explain_features_shap

__all__ = [
    "evaluate_anomaly_detector",
    "evaluate_classifier",
    "compute_partial_auc",
    "bootstrap_metric_ci",
    "evaluate_snr_robustness",
    "generate_residual_explanation",
    "explain_features_shap",
]
