"""
Baseline models for acoustic anomaly detection and fault classification.
Trained strictly on healthy baselines for anomaly models, and labeled sessions for classifiers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import xgboost as xgb


class BaseAnomalyDetector(ABC):
    """Abstract base class for all unsupervised anomaly detection models."""

    @abstractmethod
    def fit(self, X_healthy: np.ndarray) -> "BaseAnomalyDetector":
        """Fits the anomaly detector strictly on healthy baseline data."""
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns continuous anomaly score for each sample.
        Higher values indicate higher likelihood of anomaly/fault.
        """
        pass

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Binary prediction: 1 = anomaly, 0 = healthy."""
        scores = self.score_samples(X)
        return (scores > threshold).astype(int)


class HeuristicThresholdDetector(BaseAnomalyDetector):
    """Rung 1: Simple physical threshold on RMS and Kurtosis."""

    def __init__(self, rms_idx: int = 0, kurtosis_idx: int = 3):
        self.rms_idx = rms_idx
        self.kurtosis_idx = kurtosis_idx
        self.rms_p95 = 0.0
        self.kurt_p95 = 0.0

    def fit(self, X_healthy: np.ndarray) -> "HeuristicThresholdDetector":
        self.rms_p95 = float(np.percentile(X_healthy[:, self.rms_idx], 95))
        self.kurt_p95 = float(np.percentile(X_healthy[:, self.kurtosis_idx], 95))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        rms_ratio = X[:, self.rms_idx] / (self.rms_p95 + 1e-6)
        kurt_ratio = X[:, self.kurtosis_idx] / (self.kurt_p95 + 1e-6)
        # Combined score is maximum deviation normalized
        scores = np.maximum(rms_ratio, kurt_ratio)
        return scores


class PCAReconstructionDetector(BaseAnomalyDetector):
    """Rung 2A: PCA Reconstruction Error Anomaly Detector."""

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, X_healthy: np.ndarray) -> "PCAReconstructionDetector":
        X_scaled = self.scaler.fit_transform(X_healthy)
        n_comp = min(self.n_components, X_healthy.shape[0], X_healthy.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_proj = self.pca.transform(X_scaled)
        X_recon = self.pca.inverse_transform(X_proj)
        # Mean squared reconstruction error
        recon_error = np.mean((X_scaled - X_recon) ** 2, axis=1)
        return recon_error


class IsolationForestDetector(BaseAnomalyDetector):
    """Rung 2B: Isolation Forest Anomaly Detector."""

    def __init__(self, n_estimators: int = 150, contamination: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit(self, X_healthy: np.ndarray) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X_healthy)
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # score_samples returns opposite of anomaly score (lower is more anomalous)
        # Invert so higher = more anomalous
        raw_scores = -self.model.score_samples(X_scaled)
        return raw_scores


class OneClassSVMDetector(BaseAnomalyDetector):
    """Rung 2C: One-Class Support Vector Machine."""

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def fit(self, X_healthy: np.ndarray) -> "OneClassSVMDetector":
        X_scaled = self.scaler.fit_transform(X_healthy)
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # decision_function is positive for inliers, negative for outliers
        dec = self.model.decision_function(X_scaled)
        return -dec


class RandomForestFaultClassifier:
    """Rung 3A: Random Forest Multi-class Fault Diagnostic Classifier."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 12, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestFaultClassifier":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class XGBoostFaultClassifier:
    """Rung 3B: XGBoost Gradient Boosted Classifier."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 6, learning_rate: float = 0.08, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="mlogloss",
        )
        self.label_map = {}
        self.inv_label_map = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostFaultClassifier":
        X_scaled = self.scaler.fit_transform(X)
        unique_labels = sorted(list(set(y)))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.inv_label_map = {i: lbl for i, lbl in enumerate(unique_labels)}
        
        y_encoded = np.array([self.label_map[lbl] for lbl in y])
        self.model.fit(X_scaled, y_encoded)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds_enc = self.model.predict(X_scaled)
        return np.array([self.inv_label_map[i] for i in preds_enc])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
