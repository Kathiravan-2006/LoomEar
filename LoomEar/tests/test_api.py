"""Integration test for FastAPI diagnostic endpoints."""

from io import BytesIO
import numpy as np
from fastapi.testclient import TestClient
import soundfile as sf

from src.loomear.serve.api import create_app


def test_api_health_and_score_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Test Health
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 2. Test Model info
    resp = client.get("/model")
    assert resp.status_code == 200
    assert "thresholds" in resp.json()

    # 3. Test Audio Scoring
    for sr in (16000, 8000):
        dummy_wav = np.random.randn(sr * 2).astype(np.float32)
        buf = BytesIO()
        sf.write(buf, dummy_wav, sr, format="WAV")
        buf.seek(0)

        resp = client.post(
            "/score",
            files={"file": ("test.wav", buf, "audio/wav")},
            data={"machine_id": "LOOM-01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["duration_sec"] == 2.0
        assert "health_index" in data
        assert "operational_state" in data
        assert "explanation" in data
