"""Test verifying deterministic reproducibility across pipeline runs with identical random seeds."""

import numpy as np
from src.loomear.data.simulator import SyntheticLoomSimulator


def test_simulation_reproducibility():
    sim1 = SyntheticLoomSimulator(sample_rate=16000, default_rpm=210.0, random_seed=123)
    audio1, _ = sim1.generate_healthy_loom(duration_sec=2.0)

    sim2 = SyntheticLoomSimulator(sample_rate=16000, default_rpm=210.0, random_seed=123)
    audio2, _ = sim2.generate_healthy_loom(duration_sec=2.0)

    assert np.allclose(audio1, audio2, atol=1e-7), "Seed-based generation must be perfectly reproducible!"
