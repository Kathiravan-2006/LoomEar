"""Unit tests for data loaders, audio file parsing, and schema validation."""

import numpy as np
import pandas as pd
import pytest
from src.loomear.data.loaders import validate_metadata, AudioSample


def test_metadata_validation_valid():
    df = pd.DataFrame([{
        "file_id": "test_001",
        "source_type": "synthetic",
        "machine_id": "LOOM-01",
        "loom_type": "rapier",
        "rpm": 210.0,
        "session_id": "SESS-01",
        "date": "2026-03-10",
        "shift": "Shift_A",
        "mic_position": "35cm_drive",
        "condition": "healthy",
        "fault_type": "none",
        "label_source": "injected",
        "snr_db": 15.0,
        "notes": "test note",
    }])
    val_df = validate_metadata(df)
    assert len(val_df) == 1


def test_metadata_validation_duplicate_id():
    df = pd.DataFrame([
        {
            "file_id": "dup_001",
            "source_type": "synthetic",
            "machine_id": "LOOM-01",
            "loom_type": "rapier",
            "rpm": 210.0,
            "session_id": "SESS-01",
            "date": "2026-03-10",
            "shift": "Shift_A",
            "mic_position": "35cm",
            "condition": "healthy",
            "fault_type": "none",
            "label_source": "injected",
            "snr_db": 15.0,
            "notes": "",
        },
        {
            "file_id": "dup_001",
            "source_type": "synthetic",
            "machine_id": "LOOM-02",
            "loom_type": "rapier",
            "rpm": 210.0,
            "session_id": "SESS-02",
            "date": "2026-03-10",
            "shift": "Shift_A",
            "mic_position": "35cm",
            "condition": "healthy",
            "fault_type": "none",
            "label_source": "injected",
            "snr_db": 15.0,
            "notes": "",
        },
    ])
    with pytest.raises(ValueError, match="Duplicate file_id"):
        validate_metadata(df)
