"""
Maintenance decision logic with 3-state hysteresis and threshold calibration.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class LoomState(str, Enum):
    HEALTHY = "Healthy"
    WATCH = "Watch"
    ACT = "Act"


def calibrate_decision_thresholds(
    healthy_val_scores: np.ndarray,
    target_false_alarm_rate: float = 0.03,
) -> Dict[str, float]:
    """
    Calibrates operational thresholds on held-out validation data to strictly bound false alarms.
    """
    watch_threshold = float(np.percentile(healthy_val_scores, (1.0 - target_false_alarm_rate * 2.5) * 100.0))
    act_threshold = float(np.percentile(healthy_val_scores, (1.0 - target_false_alarm_rate) * 100.0))
    
    return {
        "watch_score_threshold": watch_threshold,
        "act_score_threshold": act_threshold,
        "watch_index_threshold": 60.0,
        "act_index_threshold": 80.0,
        "target_false_alarm_rate": target_false_alarm_rate,
    }


class MaintenanceDecisionEngine:
    """
    Hysteresis decision engine that requires N consecutive cycles in a state
    before changing the operational alert status, preventing alarm chattering.
    """

    def __init__(
        self,
        watch_threshold: float = 60.0,
        act_threshold: float = 80.0,
        hysteresis_consecutive: int = 4,
    ):
        self.watch_threshold = watch_threshold
        self.act_threshold = act_threshold
        self.hysteresis_consecutive = hysteresis_consecutive

        self.current_state = LoomState.HEALTHY
        self.state_candidate = LoomState.HEALTHY
        self.consecutive_count = 0

    def evaluate(self, health_index: float) -> Tuple[LoomState, bool]:
        """
        Evaluates current smoothed health index and returns (current_state, state_changed_bool).
        """
        if health_index >= self.act_threshold:
            candidate = LoomState.ACT
        elif health_index >= self.watch_threshold:
            candidate = LoomState.WATCH
        else:
            candidate = LoomState.HEALTHY

        state_changed = False
        if candidate == self.state_candidate:
            self.consecutive_count += 1
            if self.consecutive_count >= self.hysteresis_consecutive and self.current_state != candidate:
                self.current_state = candidate
                state_changed = True
        else:
            self.state_candidate = candidate
            self.consecutive_count = 1

        return self.current_state, state_changed

    def reset(self):
        """Resets the state engine."""
        self.current_state = LoomState.HEALTHY
        self.state_candidate = LoomState.HEALTHY
        self.consecutive_count = 0
