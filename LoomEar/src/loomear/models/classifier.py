"""
Diagnostic multi-class fault classifier with probability calibration.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class LoomFaultClassifier:
    """
    Rung 5: Gradient-boosted multi-class fault classifier for diagnostic root-cause isolation.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 5, learning_rate: float = 0.1, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.classes_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoomFaultClassifier":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_top_prediction(self, x: np.ndarray) -> Tuple[str, float]:
        """Returns the most probable fault class and associated confidence."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.predict_proba(x)[0]
        top_idx = int(np.argmax(probs))
        return self.classes_[top_idx], float(probs[top_idx])
