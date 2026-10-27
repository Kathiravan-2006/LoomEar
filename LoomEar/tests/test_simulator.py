"""Unit tests for synthetic loom sound generation and physical fault injection."""

import numpy as np
from src.loomear.data.simulator import SyntheticLoomSimulator


def test_simulator_healthy_generation():
    sim = SyntheticLoomSimulator(sample_rate=16000, default_rpm=210.0, random_seed=42)
    audio, cycle_starts = sim.generate_healthy_loom(duration_sec=3.0, rpm=210.0)

    assert len(audio) == 3 * 16000
    assert len(cycle_starts) >= 8  # 210 rpm = 3.5 picks/sec -> ~10 picks in 3s
    assert np.max(np.abs(audio)) > 0.01
    assert not np.isnan(audio).any()


def test_simulator_fault_injection():
    sim = SyntheticLoomSimulator(sample_rate=16000, random_seed=42)
    clean, _ = sim.generate_healthy_loom(duration_sec=2.0, rpm=200.0)

    faults = ["bearing_wear", "timing_drift", "belt_squeal", "lubrication_loss", "beatup_looseness", "loose_frame"]
    for f in faults:
        corrupted = sim.inject_fault(clean, fault_type=f, severity=0.7, rpm=200.0)
        assert len(corrupted) == len(clean)
        assert not np.isnan(corrupted).any()
        # Verify signal was actually modified
        assert not np.allclose(clean, corrupted)
