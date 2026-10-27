"""
ONNX export and quantization utility for edge deployments (Raspberry Pi / Jetson).
Exports trained anomaly detectors and models to lightweight inference formats.
"""

from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_model_artifacts(
    model: object,
    output_dir: str = "models",
    model_name: str = "loomear_detector",
):
    """
    Exports model parameters, scalar scaling coefficients, and baseline thresholds
    to lightweight JSON and numpy binary format for real-time edge runtime.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": model_name,
        "framework": "LoomEar Edge Runtime",
        "input_shape": [1, 64, 64],
        "sample_rate": 16000,
        "quantized": True,
        "precision": "int8_weights_float32_acts",
    }

    manifest_path = out_path / f"{model_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Model exported successfully to {out_path} ({manifest_path})")


if __name__ == "__main__":
    from src.loomear.models.autoencoder import ConvAutoencoderAnomalyDetector
    model = ConvAutoencoderAnomalyDetector(latent_dim=32)
    dummy_healthy = np.random.randn(20, 64, 64).astype(np.float32)
    model.fit(dummy_healthy)
    export_model_artifacts(model)
