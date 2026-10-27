"""
Industrial Maintenance Cost Simulator and ROI Calculator for Textile Mills.
Models downtime economics, sensor hardware costs, false-alarm overheads, and net savings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class MaintenanceCostModel:
    """
    Transparent predictive maintenance economic simulator.
    """

    def __init__(
        self,
        downtime_cost_per_hour_usd: float = 120.0,
        unplanned_downtime_hours_per_year: float = 75.0,
        early_detection_rate: float = 0.78,
        false_alarm_investigation_cost_usd: float = 25.0,
        false_alarms_per_year: float = 4.0,
        annual_sensor_edge_cost_usd: float = 65.0,
        num_looms: int = 24,
    ):
        self.downtime_cost = downtime_cost_per_hour_usd
        self.unplanned_hours = unplanned_downtime_hours_per_year
        self.detection_rate = early_detection_rate
        self.false_alarm_cost = false_alarm_investigation_cost_usd
        self.false_alarms_year = false_alarms_per_year
        self.sensor_cost = annual_sensor_edge_cost_usd
        self.num_looms = num_looms

    def calculate_single_loom(self) -> Dict[str, float]:
        """Calculates annual economic figures for a single power loom."""
        gross_downtime_loss = self.unplanned_hours * self.downtime_cost
        prevented_downtime_savings = self.detection_rate * gross_downtime_loss
        false_alarm_overhead = self.false_alarms_year * self.false_alarm_cost
        total_costs = self.sensor_cost + false_alarm_overhead

        net_saving_per_loom = prevented_downtime_savings - total_costs
        roi_pct = (net_saving_per_loom / max(1.0, total_costs)) * 100.0

        return {
            "gross_downtime_loss_usd": gross_downtime_loss,
            "prevented_downtime_savings_usd": prevented_downtime_savings,
            "annual_sensor_cost_usd": self.sensor_cost,
            "false_alarm_overhead_usd": false_alarm_overhead,
            "net_annual_saving_usd": net_saving_per_loom,
            "roi_percent": roi_pct,
        }

    def calculate_fleet(self) -> Dict[str, float]:
        """Calculates fleet-wide economic savings."""
        single = self.calculate_single_loom()
        return {
            "num_looms": self.num_looms,
            "fleet_gross_loss_usd": single["gross_downtime_loss_usd"] * self.num_looms,
            "fleet_gross_savings_usd": single["prevented_downtime_savings_usd"] * self.num_looms,
            "fleet_total_system_cost_usd": (single["annual_sensor_cost_usd"] + single["false_alarm_overhead_usd"]) * self.num_looms,
            "fleet_net_annual_savings_usd": single["net_annual_saving_usd"] * self.num_looms,
            "roi_percent": single["roi_percent"],
        }


def calculate_mill_savings(
    downtime_cost: float,
    unplanned_hours: float,
    detection_rate: float,
    num_looms: int = 24,
) -> float:
    """Helper function to quickly calculate net annual savings for given parameters."""
    model = MaintenanceCostModel(
        downtime_cost_per_hour_usd=downtime_cost,
        unplanned_downtime_hours_per_year=unplanned_hours,
        early_detection_rate=detection_rate,
        num_looms=num_looms,
    )
    return model.calculate_fleet()["fleet_net_annual_savings_usd"]


def generate_sensitivity_matrix(
    detection_rates: Optional[List[float]] = None,
    downtime_costs: Optional[List[float]] = None,
    num_looms: int = 24,
) -> pd.DataFrame:
    """
    Generates a 2D sensitivity grid varying Early Detection Rate vs Hourly Downtime Cost.
    """
    if detection_rates is None:
        detection_rates = [0.50, 0.60, 0.70, 0.80, 0.90]
    if downtime_costs is None:
        downtime_costs = [60.0, 90.0, 120.0, 150.0, 200.0]

    grid = {}
    for cost in downtime_costs:
        col_name = f"${int(cost)}/hr"
        grid[col_name] = []
        for rate in detection_rates:
            savings = calculate_mill_savings(cost, 75.0, rate, num_looms=num_looms)
            grid[col_name].append(round(savings, 2))

    df = pd.DataFrame(grid, index=[f"{int(r*100)}% Catch Rate" for r in detection_rates])
    return df
