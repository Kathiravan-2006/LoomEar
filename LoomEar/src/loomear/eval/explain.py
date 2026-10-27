"""
Explainability engine: Residual heatmaps and SHAP feature importance.
Provides plain-English interpretations for mill mechanics and plant managers.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def generate_residual_explanation(
    residual_map: np.ndarray,
    freq_bands: Tuple[float, float] = (50.0, 7800.0),
    top_k: int = 3,
) -> Dict[str, object]:
    """
    Analyzes cycle residual map and produces an intuitive plain-English fault diagnostic statement.
    """
    n_mels, n_time = residual_map.shape
    
    # Frequency partition
    low_band_res = np.mean(residual_map[: int(n_mels * 0.25), :] ** 2)
    mid_band_res = np.mean(residual_map[int(n_mels * 0.25) : int(n_mels * 0.65), :] ** 2)
    high_band_res = np.mean(residual_map[int(n_mels * 0.65) :, :] ** 2)

    # Time partition: Early (0-30%), Mid (30-60%), Late (60-100%)
    t_early = np.mean(residual_map[:, : int(n_time * 0.30)] ** 2)
    t_mid = np.mean(residual_map[:, int(n_time * 0.30) : int(n_time * 0.60)] ** 2)
    t_late = np.mean(residual_map[:, int(n_time * 0.60) :] ** 2)

    dominant_band = "High-Frequency (3.5-7.8 kHz)"
    if low_band_res > high_band_res and low_band_res > mid_band_res:
        dominant_band = "Low-Frequency (50-500 Hz)"
    elif mid_band_res > high_band_res:
        dominant_band = "Mid-Frequency (500-3000 Hz)"

    dominant_phase = "Mid-Pick Beat-Up Impact"
    if t_early > t_mid and t_early > t_late:
        dominant_phase = "Early Pick Shedding Motion"
    elif t_late > t_mid:
        dominant_phase = "Late Pick Weft Insertion / Cut"

    explanation = (
        f"Acoustic deviation concentrated in the {dominant_band} band during the {dominant_phase} phase. "
        f"Energy anomaly is {round(max(low_band_res, mid_band_res, high_band_res), 2)}x baseline."
    )

    return {
        "dominant_band": dominant_band,
        "dominant_phase": dominant_phase,
        "band_energies": {
            "low_band_50_500hz": float(low_band_res),
            "mid_band_500_3000hz": float(mid_band_res),
            "high_band_3000_7800hz": float(high_band_res),
        },
        "phase_energies": {
            "early_shedding": float(t_early),
            "mid_beatup": float(t_mid),
            "late_weft_cut": float(t_late),
        },
        "plain_english_summary": explanation,
    }


def explain_features_shap(
    model: object,
    X_train: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Computes SHAP feature importance for tree models.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # multi-class: take mean across classes
            shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_abs = np.abs(shap_values)

        mean_importance = np.mean(shap_abs, axis=0)
        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, mean_importance)
        }
        # Sort descending
        sorted_imp = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_imp
    except Exception:
        # Fallback to feature importance from model if SHAP encounters issue
        if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            imps = model.model.feature_importances_
            d = {name: float(imp) for name, imp in zip(feature_names, imps)}
            return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        return {name: 1.0 / len(feature_names) for name in feature_names[:10]}
