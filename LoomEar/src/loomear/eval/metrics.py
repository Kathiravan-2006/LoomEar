"""
Evaluation metrics computation, partial AUC, confusion matrices, and bootstrap confidence intervals.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_partial_auc(y_true: np.ndarray, y_scores: np.ndarray, max_fpr: float = 0.10) -> float:
    """
    Computes partial ROC AUC strictly over low false positive rates (FPR <= max_fpr).
    Crucial for industrial predictive maintenance where high false alarm rates are impermissible.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # Restrict to FPR <= max_fpr
        mask = fpr <= max_fpr
        if not np.any(mask) or np.sum(mask) < 2:
            return 0.5
        fpr_sub = np.concatenate([fpr[mask], [max_fpr]])
        tpr_sub = np.concatenate([tpr[mask], [np.interp(max_fpr, fpr, tpr)]])
        p_auc = float(np.trapz(tpr_sub, fpr_sub) / max_fpr)
        return float(np.clip(p_auc, 0.0, 1.0))
    except Exception:
        return 0.5


def evaluate_anomaly_detector(
    y_true_binary: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluates unsupervised anomaly detection performance:
    ROC AUC, Partial AUC (FPR <= 0.10), Precision, Recall, and F1.
    """
    if threshold is None:
        threshold = float(np.percentile(anomaly_scores[y_true_binary == 0], 95)) if np.any(y_true_binary == 0) else float(np.median(anomaly_scores))

    y_pred = (anomaly_scores > threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true_binary, anomaly_scores))
    except Exception:
        roc_auc = 0.5

    p_auc = compute_partial_auc(y_true_binary, anomaly_scores, max_fpr=0.10)
    precision = float(precision_score(y_true_binary, y_pred, zero_division=0))
    recall = float(recall_score(y_true_binary, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred, zero_division=0))

    return {
        "roc_auc": round(roc_auc, 4),
        "partial_auc_0.10": round(p_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 4),
    }


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict[str, float], List[List[int]]]]:
    """
    Evaluates multi-class fault classification performance:
    Macro F1, Accuracy, per-class recall, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Per-class recall
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_class_rec = {cls: round(float(rec), 4) for cls, rec in zip(classes, recalls)}

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_recall": per_class_rec,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_func=roc_auc_score,
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes non-parametric bootstrap confidence interval: (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_metrics = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            val = float(metric_func(y_true[idx], scores[idx]))
            boot_metrics.append(val)
        except Exception:
            continue

    if not boot_metrics:
        return 0.5, 0.5, 0.5

    mean_val = float(np.mean(boot_metrics))
    alpha = (1.0 - confidence_level) / 2.0
    lower_ci = float(np.percentile(boot_metrics, alpha * 100))
    upper_ci = float(np.percentile(boot_metrics, (1.0 - alpha) * 100))

    return round(mean_val, 4), round(lower_ci, 4), round(upper_ci, 4)
