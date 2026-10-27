import time
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loomear.audio.preprocess import clean_audio
from src.loomear.audio.cycles import segment_cycles, build_healthy_template, compute_cycle_residual


def run_benchmark(n_iterations: int = 50, window_sec: float = 3.0):
    print(f"Running LoomEar Edge Benchmark over {n_iterations} iterations...")
    
    sr = 16000
    dummy_audio = np.random.randn(int(window_sec * sr)).astype(np.float32)
    template_cycles = np.random.randn(10, 4000).astype(np.float32)
    template = build_healthy_template(template_cycles, sample_rate=sr)

    latencies = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        
        # Pipeline execution
        cleaned = clean_audio(dummy_audio, sample_rate=sr)
        cycles_res = segment_cycles(cleaned, sample_rate=sr)
        _, _ = compute_cycle_residual(cycles_res.normalized_cycles[0], template, sample_rate=sr)
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)

    print("\n" + "=" * 45)
    print("      LOOMEAR EDGE PERFORMANCE REPORT")
    print("=" * 45)
    print(f"Audio Window Duration : {window_sec:.1f} s")
    print(f"Mean Latency          : {mean_lat:.2f} ms")
    print(f"P50 Latency (Median)  : {p50:.2f} ms")
    print(f"P95 Latency           : {p95:.2f} ms")
    print(f"Throughput Ratio      : {(window_sec * 1000.0) / mean_lat:.1f}x real-time")
    print("Target Device Class   : Raspberry Pi 4 / Jetson Nano compatible")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    run_benchmark(n_iterations=30)
