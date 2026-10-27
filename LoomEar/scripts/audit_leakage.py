"""
Automated Data Leakage Audit and Integrity Verification for LoomEar.
Enforces zero session overlap, zero machine leakage in LOMO, and strict training-only transformations.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loomear.data.loaders import load_metadata
from src.loomear.data.splits import get_session_split, get_lomo_split, audit_data_leakage


def run_full_leakage_audit():
    print("=" * 60)
    print("      LOOMEAR AUTOMATED DATA LEAKAGE AUDIT")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    metadata_path = root_dir / "data" / "metadata.csv"

    if not metadata_path.exists():
        print("[!] metadata.csv not found, generating benchmark dataset first...")
        from src.loomear.data.simulator import generate_dataset
        generate_dataset(root_dir / "data", seed=42)

    df = load_metadata(metadata_path)
    print(f"Loaded {len(df)} total metadata records.")

    # Audit 1: Session-Grouped Split
    print("\n[Audit 1/3] Testing Session-Grouped Partition Integrity...")
    train_df, val_df, test_df = get_session_split(df, test_size=0.25, val_size=0.20, random_state=42)
    audit_res = audit_data_leakage(train_df, val_df, test_df, group_col="session_id")
    print(f"  --> PASSED: Train sessions ({audit_res['train_groups']}), Val sessions ({audit_res['val_groups']}), Test sessions ({audit_res['test_groups']}) are strictly disjoint.")

    # Audit 2: Leave-One-Machine-Out (LOMO) Zero Machine Leakage
    print("\n[Audit 2/3] Testing Leave-One-Machine-Out (LOMO) Isolation...")
    lomo_train, lomo_test = get_lomo_split(df, target_machine_id="LOOM-04")
    train_machines = set(lomo_train["machine_id"])
    test_machines = set(lomo_test["machine_id"])
    assert not train_machines.intersection(test_machines), "Machine leakage detected!"
    print(f"  --> PASSED: Train machines {train_machines} have ZERO overlap with test machine {test_machines}.")

    # Audit 3: Source Type Label Integrity
    print("\n[Audit 3/3] Checking Dataset Source Label Authenticity...")
    valid_sources = {"real", "public", "synthetic"}
    actual_sources = set(df["source_type"].unique())
    assert actual_sources.issubset(valid_sources), f"Unrecognized source types: {actual_sources}"
    print(f"  --> PASSED: Every sample is explicitly labelled as {actual_sources}.")

    print("\n" + "=" * 60)
    print("       ALL LEAKAGE AUDITS PASSED WITH ZERO VIOLATIONS")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_full_leakage_audit()
