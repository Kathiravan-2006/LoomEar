"""
Session-grouped and machine-grouped data splitting strategies with strict leakage prevention.
"""

from typing import Dict, List, Set, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

logger = logging.getLogger(__name__)


def get_session_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata by session_id so all recordings from any given session
    appear exclusively in Train, Validation, or Test.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["session_id"]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Split train_val into train and val
    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["session_id"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Audit for session leakage
    audit_data_leakage(train_df, val_df, test_df, group_col="session_id")

    logger.info(
        f"Session Split - Train: {len(train_df)} ({train_df['session_id'].nunique()} sessions), "
        f"Val: {len(val_df)} ({val_df['session_id'].nunique()} sessions), "
        f"Test: {len(test_df)} ({test_df['session_id'].nunique()} sessions)"
    )

    return train_df, val_df, test_df


def get_lomo_split(
    df: pd.DataFrame,
    target_machine_id: str = "LOOM-04",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-One-Machine-Out (LOMO) split: Trains strictly on all other machines
    and reserves the target machine exclusively for zero-shot testing.
    """
    if target_machine_id not in df["machine_id"].values:
        raise ValueError(f"Target machine '{target_machine_id}' not found in metadata.")

    test_df = df[df["machine_id"] == target_machine_id].copy()
    train_df = df[df["machine_id"] != target_machine_id].copy()

    # Audit for machine leakage
    train_machines = set(train_df["machine_id"])
    test_machines = set(test_df["machine_id"])
    overlap = train_machines.intersection(test_machines)
    if overlap:
        raise ValueError(f"Machine leakage detected! Overlapping machines: {overlap}")

    logger.info(
        f"LOMO Split ({target_machine_id}) - Train: {len(train_df)} samples, "
        f"Held-out Machine Test: {len(test_df)} samples."
    )

    return train_df, test_df


def audit_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "session_id",
) -> Dict[str, bool]:
    """
    Formally verifies that no recording sessions, file IDs, or machine identities leak across partitions.
    Raises ValueError on any violation.
    """
    train_groups: Set[str] = set(train_df[group_col])
    val_groups: Set[str] = set(val_df[group_col])
    test_groups: Set[str] = set(test_df[group_col])

    train_val_leak = train_groups.intersection(val_groups)
    train_test_leak = train_groups.intersection(test_groups)
    val_test_leak = val_groups.intersection(test_groups)

    train_files = set(train_df["file_id"])
    val_files = set(val_df["file_id"])
    test_files = set(test_df["file_id"])

    file_leaks = train_files.intersection(val_files) | train_files.intersection(test_files) | val_files.intersection(test_files)

    if train_val_leak or train_test_leak or val_test_leak or file_leaks:
        error_msg = (
            f"CRITICAL DATA LEAKAGE DETECTED!\n"
            f"Train/Val Overlap ({group_col}): {train_val_leak}\n"
            f"Train/Test Overlap ({group_col}): {train_test_leak}\n"
            f"Val/Test Overlap ({group_col}): {val_test_leak}\n"
            f"File ID Leaks: {file_leaks}\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return {
        "passed": True,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
    }
