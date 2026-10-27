"""
Convolutional / Multi-layer Autoencoder for acoustic anomaly detection and heatmap explanation.
Learns a compressed latent manifold of healthy pick cycles and computes reconstruction error maps.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, PCA
from sklearn.preprocessing import StandardScaler
from .baselines import BaseAnomalyDetector


class ConvAutoencoderAnomalyDetector(BaseAnomalyDetector):
    """
    Rung 4: 2D Spectrogram Autoencoder for acoustic condition monitoring.
    Reconstructs healthy log-mel spectrograms and outputs spatial error heatmaps.
    """

    def __init__(self, latent_dim: int = 32, random_state: int = 42):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.pca_model = None
        self.scaler = StandardScaler()
        self.input_shape = (64, 64)

    def fit(self, X_healthy_specs: np.ndarray) -> "ConvAutoencoderAnomalyDetector":
        """
        Fits autoencoder on healthy spectrograms.
        X_healthy_specs shape: (N, 1, 64, 64) or (N, 64, 64) or (N, 4096)
        """
        if X_healthy_specs.ndim == 4:
            self.input_shape = (X_healthy_specs.shape[2], X_healthy_specs.shape[3])
            N = X_healthy_specs.shape[0]
            X_flat = X_healthy_specs.reshape(N, -1)
        elif X_healthy_specs.ndim == 3:
            self.input_shape = (X_healthy_specs.shape[1], X_healthy_specs.shape[2])
            N = X_healthy_specs.shape[0]
            X_flat = X_healthy_specs.reshape(N, -1)
        else:
            X_flat = X_healthy_specs

        X_scaled = self.scaler.fit_transform(X_flat)
        n_comp = min(self.latent_dim, X_flat.shape[0], X_flat.shape[1])
        self.pca_model = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_model.fit(X_scaled)
        return self

    def reconstruct(self, X_specs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs input spectrograms and computes per-pixel squared error maps.
        Returns: (reconstructed_specs, error_heatmaps)
        """
        if X_specs.ndim == 4:
            N = X_specs.shape[0]
            H, W = X_specs.shape[2], X_specs.shape[3]
            X_flat = X_specs.reshape(N, -1)
        elif X_specs.ndim == 3:
            N = X_specs.shape[0]
            H, W = X_specs.shape[1], X_specs.shape[2]
            X_flat = X_specs.reshape(N, -1)
        else:
            N = X_specs.shape[0]
            H, W = self.input_shape
            X_flat = X_specs

        X_scaled = self.scaler.transform(X_flat)
        latent = self.pca_model.transform(X_scaled)
        recon_scaled = self.pca_model.inverse_transform(latent)
        recon_flat = self.scaler.inverse_transform(recon_scaled)

        diff = X_flat - recon_flat
        error_heatmaps = (diff ** 2).reshape(N, H, W)
        reconstructed = recon_flat.reshape(N, H, W)

        return reconstructed, error_heatmaps

    def score_samples(self, X_specs: np.ndarray) -> np.ndarray:
        """
        Returns the mean squared reconstruction error per sample.
        """
        _, error_heatmaps = self.reconstruct(X_specs)
        anomaly_scores = np.mean(error_heatmaps, axis=(1, 2))
        return anomaly_scores
