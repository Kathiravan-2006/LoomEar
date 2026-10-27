"""Unit tests for audio filtering, pick-cycle segmentation, and residual computation."""

import numpy as np
from src.loomear.audio.preprocess import clean_audio, bandpass_filter, remove_dc
from src.loomear.audio.cycles import (
    estimate_pick_period,
    segment_cycles,
    build_healthy_template,
    compute_cycle_residual,
    time_normalize_cycle,
)
from src.loomear.data.simulator import SyntheticLoomSimulator


def test_bandpass_and_dc_removal():
    sr = 16000
    t = np.arange(sr * 2) / sr
    # 10 Hz sub-rumble + 1000 Hz tone + DC bias of 2.0
    sig = 2.0 + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 1000 * t)
    
    dc_free = remove_dc(sig)
    assert np.abs(np.mean(dc_free)) < 1e-6

    cleaned = clean_audio(sig, sample_rate=sr, low_hz=50.0, high_hz=7800.0)
    assert np.abs(np.mean(cleaned)) < 0.02
    assert np.max(np.abs(cleaned)) <= 1.0


def test_pick_period_estimation_accuracy():
    sim = SyntheticLoomSimulator(sample_rate=16000, random_seed=42)
    target_rpm = 210.0
    audio, _ = sim.generate_healthy_loom(duration_sec=6.0, rpm=target_rpm)
    cleaned = clean_audio(audio, sample_rate=16000)

    est_rpm, est_period, conf = estimate_pick_period(cleaned, sample_rate=16000)
    assert abs(est_rpm - target_rpm) < 5.0
    assert conf > 0.40


def test_cycle_segmentation_and_template():
    sim = SyntheticLoomSimulator(sample_rate=16000, random_seed=42)
    audio, _ = sim.generate_healthy_loom(duration_sec=4.0, rpm=210.0)
    cleaned = clean_audio(audio, sample_rate=16000)

    res = segment_cycles(cleaned, sample_rate=16000, target_samples=4000)
    assert len(res.cycles) > 5
    assert res.normalized_cycles.shape[1] == 4000

    template = build_healthy_template(res.normalized_cycles, sample_rate=16000)
    assert "mean_spec" in template
    assert "std_spec" in template

    # Residual on self should be small
    res_map, anomaly_score = compute_cycle_residual(res.normalized_cycles[0], template, sample_rate=16000)
    assert anomaly_score < 2.0
