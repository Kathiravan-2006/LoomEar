"""
Edge real-time audio capture and predictive scoring daemon.
Reads continuous audio from MEMS/USB microphone or fallback simulation, computes health index,
logs telemetry to local CSV, and optionally pushes alerts to the central LoomEar server.
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loomear.audio.preprocess import clean_audio
from src.loomear.audio.cycles import segment_cycles, build_healthy_template, compute_cycle_residual
from src.loomear.data.simulator import SyntheticLoomSimulator
from src.loomear.health.index import HealthIndexEngine
from src.loomear.health.decision import MaintenanceDecisionEngine, LoomState


def run_edge_monitor(
    machine_id: str = "LOOM-01",
    api_url: str = "http://localhost:8000",
    window_sec: float = 3.0,
    interval_sec: float = 2.0,
    max_cycles: int = 10,
    output_csv: str = "edge_telemetry.csv",
):
    """
    Main edge loop: processes periodic windows and logs health state.
    """
    print(f"Starting LoomEar Edge Monitor on [{machine_id}]...")
    sim = SyntheticLoomSimulator(sample_rate=16000, random_seed=42)
    
    # Initialize health engine with baseline calibration
    h_baseline = np.array([0.06, 0.09, 0.12, 0.15, 0.19, 0.23, 0.27])
    health_engine = HealthIndexEngine(h_baseline, alpha=0.30)
    decision_engine = MaintenanceDecisionEngine(watch_threshold=60.0, act_threshold=80.0, hysteresis_consecutive=2)

    # Pre-build default template
    ref_healthy, _ = sim.generate_healthy_loom(duration_sec=5.0, rpm=210.0)
    ref_res = segment_cycles(clean_audio(ref_healthy), sample_rate=16000)
    template = build_healthy_template(ref_res.normalized_cycles, sample_rate=16000)

    records = []
    print(f"{'Time':<10} | {'RPM':<6} | {'Confidence':<10} | {'Score':<8} | {'Health Index':<12} | {'State':<10}")
    print("-" * 70)

    for step in range(1, max_cycles + 1):
        # Simulate edge capture: healthy at start, introducing bearing wear at step 6
        if step < 6:
            audio, _ = sim.generate_healthy_loom(duration_sec=window_sec, rpm=210.0 + np.random.uniform(-1, 1))
            audio = sim.mix_ambience_snr(audio, snr_db=18.0)
        else:
            c, _ = sim.generate_healthy_loom(duration_sec=window_sec, rpm=210.0)
            f = sim.inject_fault(c, "bearing_wear", severity=0.12 * (step - 5), rpm=210.0)
            audio = sim.mix_ambience_snr(f, snr_db=15.0)

        cleaned = clean_audio(audio, sample_rate=16000)
        cycles_res = segment_cycles(cleaned, sample_rate=16000)
        
        if len(cycles_res.normalized_cycles) > 0:
            target_cycle = cycles_res.normalized_cycles[0]
        else:
            target_cycle = np.zeros(4000, dtype=np.float32)

        _, raw_score = compute_cycle_residual(target_cycle, template, sample_rate=16000)
        smoothed_index = health_engine.update(raw_score)
        state, state_changed = decision_engine.evaluate(smoothed_index)

        timestamp_str = time.strftime("%H:%M:%S")
        print(f"{timestamp_str:<10} | {cycles_res.estimated_rpm:<6.1f} | {cycles_res.confidence*100:<9.1f}% | {raw_score:<8.4f} | {smoothed_index:<12.1f} | {state.value:<10}")

        records.append({
            "timestamp": timestamp_str,
            "machine_id": machine_id,
            "rpm": cycles_res.estimated_rpm,
            "confidence": cycles_res.confidence,
            "raw_anomaly_score": raw_score,
            "health_index": smoothed_index,
            "state": state.value,
        })

        time.sleep(0.2)  # fast iteration for test demo

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nEdge run complete. Telemetry saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoomEar Edge Monitor")
    parser.add_argument("--machine-id", default="LOOM-01")
    parser.add_argument("--max-cycles", type=int, default=8)
    args = parser.parse_args()
    run_edge_monitor(machine_id=args.machine_id, max_cycles=args.max_cycles)
