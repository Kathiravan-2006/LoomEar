"""
Script to generate the complete multi-loom synthetic benchmark audio dataset.
"""

from pathlib import Path
import sys

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loomear.data.simulator import generate_dataset


def main():
    print("Generating LoomEar benchmark dataset...")
    out_dir = Path(__file__).parent.parent / "data"
    metadata_df = generate_dataset(output_dir=out_dir, seed=42)
    print(f"Dataset generated successfully! {len(metadata_df)} recordings created.")
    print(f"Metadata summary:\n{metadata_df['condition'].value_counts()}")


if __name__ == "__main__":
    main()
