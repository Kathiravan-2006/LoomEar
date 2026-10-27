"""Unit tests for health indexing, hysteresis decision logic, and cost modeling."""

import numpy as np
from src.loomear.health.index import compute_health_index, smooth_health_index
from src.loomear.health.decision import MaintenanceDecisionEngine, LoomState
from src.loomear.health.cost_model import MaintenanceCostModel


def test_health_index_and_smoothing():
    h_baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    test_scores = np.array([0.2, 0.45, 1.2])

    indices = compute_health_index(test_scores, h_baseline)
    assert indices[0] < indices[1] < indices[2]
    assert 0.0 <= indices[0] <= 100.0

    smoothed = smooth_health_index(indices, alpha=0.3)
    assert len(smoothed) == len(indices)


def test_hysteresis_decision_engine():
    engine = MaintenanceDecisionEngine(watch_threshold=60.0, act_threshold=80.0, hysteresis_consecutive=3)
    
    # Needs 3 consecutive readings above 60 to change state
    state, changed = engine.evaluate(65.0)
    assert state == LoomState.HEALTHY
    state, changed = engine.evaluate(65.0)
    assert state == LoomState.HEALTHY
    state, changed = engine.evaluate(65.0)
    assert state == LoomState.WATCH
    assert changed is True


def test_cost_model_roi():
    model = MaintenanceCostModel(downtime_cost_per_hour_usd=120.0, unplanned_downtime_hours_per_year=75.0, early_detection_rate=0.78, num_looms=24)
    res = model.calculate_fleet()
    assert res["fleet_net_annual_savings_usd"] > 50000.0
    assert res["roi_percent"] > 500.0
