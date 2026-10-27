"""Unit tests for session and machine-grouped data splitting strategies."""

import pandas as pd
import pytest
from src.loomear.data.splits import get_session_split, get_lomo_split, audit_data_leakage


def test_session_grouped_split_no_leakage():
    # Build synthetic mock metadata with 6 distinct sessions
    records = []
    for s in range(1, 7):
        for rep in range(4):
            records.append({
                "file_id": f"file_{s}_{rep}",
                "source_type": "synthetic",
                "machine_id": f"LOOM-0{1 + s%2}",
                "session_id": f"SESS-{s:02d}",
                "condition": "healthy",
                "fault_type": "none",
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = get_session_split(df, test_size=0.33, val_size=0.33, random_state=42)

    # Assert no session overlap
    train_sess = set(train_df["session_id"])
    val_sess = set(val_df["session_id"])
    test_sess = set(test_df["session_id"])

    assert len(train_sess.intersection(val_sess)) == 0
    assert len(train_sess.intersection(test_sess)) == 0
    assert len(val_sess.intersection(test_sess)) == 0


def test_lomo_split_strict_machine_isolation():
    records = []
    for m in ["LOOM-01", "LOOM-02", "LOOM-03", "LOOM-04"]:
        for rep in range(3):
            records.append({
                "file_id": f"file_{m}_{rep}",
                "source_type": "synthetic",
                "machine_id": m,
                "session_id": f"SESS-{m}",
                "condition": "healthy",
                "fault_type": "none",
            })
    df = pd.DataFrame(records)

    train_df, test_df = get_lomo_split(df, target_machine_id="LOOM-04")

    assert "LOOM-04" not in train_df["machine_id"].values
    assert set(test_df["machine_id"].unique()) == {"LOOM-04"}
