"""Unit tests for feature extraction and feature matrix shapes."""

import numpy as np
from src.loomear.features.handcrafted import extract_features_for_sample, get_feature_names


def test_handcrafted_features_shape_and_nans():
    sr = 16000
    dummy_audio = np.random.randn(sr * 2).astype(np.float32)
    feat_names = get_feature_names()

    f_dict = extract_features_for_sample(dummy_audio, sample_rate=sr)
    assert len(f_dict) == len(feat_names)

    for name, val in f_dict.items():
        assert not np.isnan(val), f"Feature {name} is NaN"
        assert not np.isinf(val), f"Feature {name} is Inf"
