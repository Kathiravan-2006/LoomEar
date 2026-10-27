"""Unit tests for anomaly detectors and baseline models."""

import numpy as np
from src.loomear.models.baselines import (
    HeuristicThresholdDetector,
    PCAReconstructionDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    RandomForestFaultClassifier,
)
from src.loomear.models.autoencoder import ConvAutoencoderAnomalyDetector


def test_anomaly_detectors_fit_and_score():
    rng = np.random.default_rng(42)
    # 20 healthy samples, 5 anomalous
    X_healthy = rng.normal(loc=0.0, scale=1.0, size=(20, 50))
    X_anomalous = rng.normal(loc=4.0, scale=2.0, size=(5, 50))
    X_test = np.vstack([X_healthy[:5], X_anomalous])

    detectors = [
        HeuristicThresholdDetector(),
        PCAReconstructionDetector(n_components=4),
        IsolationForestDetector(n_estimators=50),
        OneClassSVMDetector(),
    ]

    for det in detectors:
        det.fit(X_healthy)
        scores = det.score_samples(X_test)
        assert len(scores) == len(X_test)
        # Anomalous samples should have higher mean score than healthy test samples
        assert np.mean(scores[5:]) >= np.mean(scores[:5])


def test_autoencoder_reconstruction_shape():
    X_healthy_specs = np.random.randn(15, 64, 64).astype(np.float32)
    ae = ConvAutoencoderAnomalyDetector(latent_dim=16)
    ae.fit(X_healthy_specs)

    scores = ae.score_samples(X_healthy_specs)
    assert len(scores) == 15
    assert not np.isnan(scores).any()

    recon, heatmaps = ae.reconstruct(X_healthy_specs[:2])
    assert heatmaps.shape == (2, 64, 64)
