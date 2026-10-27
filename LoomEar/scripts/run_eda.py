"""
Exploratory Data Analysis and publication figure generation for LoomEar.
Produces 300 DPI figures with sidecar caption metadata.
"""

from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loomear.data.loaders import load_dataset, load_metadata
from src.loomear.audio.preprocess import clean_audio
from src.loomear.audio.cycles import segment_cycles, build_healthy_template, compute_cycle_residual, compute_log_mel_spectrogram
from src.loomear.features.handcrafted import extract_features_for_sample


def save_figure_with_sidecar(
    fig: plt.Figure,
    fig_path: Path,
    caption: str,
    source_type: str = "synthetic",
    units: str = "Normalized amplitude, Frequency (Hz), Time (s)",
):
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    sidecar_path = fig_path.with_suffix(".json")
    meta = {
        "figure_name": fig_path.name,
        "caption": caption,
        "source_type": source_type,
        "units": units,
        "dpi": 300,
    }
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    fig_dir = root_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        from src.loomear.data.simulator import generate_dataset
        generate_dataset(data_dir, seed=42)

    df = load_metadata(metadata_path)
    samples = load_dataset(data_dir / "raw" / "synthetic", metadata_path)

    print(f"Running EDA on {len(samples)} samples across {df['machine_id'].nunique()} machines...")

    # 1. Dataset Overview Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="condition", hue="machine_id", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Sample Distribution by Condition and Machine")
    axes[0].set_ylabel("Number of Recordings")
    axes[0].set_xlabel("Condition State")

    sns.countplot(data=df, y="fault_type", hue="condition", ax=axes[1], palette="Set2")
    axes[1].set_title("Fault Mode Breakdown")
    axes[1].set_ylabel("Injected Fault Type")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    save_figure_with_sidecar(
        fig,
        fig_dir / "01_dataset_distribution.png",
        caption="Distribution of loom acoustic dataset partitioned by operational condition, machine ID, and physical fault mode.",
        source_type="synthetic",
    )

    # 2. Waveform and Spectrogram Comparison (Healthy vs Bearing Wear vs Beat-up Looseness)
    healthy_sample = next(s for s in samples if s.metadata["condition"] == "healthy")
    bearing_sample = next(s for s in samples if s.metadata["fault_type"] == "bearing_wear")
    beatup_sample = next(s for s in samples if s.metadata["fault_type"] == "beatup_looseness")

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    pairs = [
        ("Healthy Baseline (LOOM-01)", healthy_sample),
        ("Bearing Wear Fault (High-Freq Hiss)", bearing_sample),
        ("Beat-up Looseness (Doubled Impact)", beatup_sample),
    ]

    for idx, (title, s) in enumerate(pairs):
        t = np.arange(len(s.audio)) / s.sample_rate
        axes[idx, 0].plot(t[:16000*2], s.audio[:16000*2], color="#1E3A8A", lw=0.8)
        axes[idx, 0].set_ylabel(title, fontsize=9, fontweight="bold")
        axes[idx, 0].set_ylim(-1.0, 1.0)
        axes[idx, 0].grid(True, alpha=0.3)

        spec = compute_log_mel_spectrogram(s.audio[:16000*2], sample_rate=16000)
        axes[idx, 1].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[idx, 1].set_ylabel("Mel Bins")

    axes[2, 0].set_xlabel("Time (seconds)")
    axes[2, 1].set_xlabel("Time Frames")
    axes[0, 0].set_title("Time-Domain Acoustic Waveform")
    axes[0, 1].set_title("Log-Mel Spectrogram")
    plt.tight_layout()
    save_figure_with_sidecar(
        fig,
        fig_dir / "02_waveform_spectrogram_comparison.png",
        caption="Comparative waveforms and log-mel spectrograms for healthy baseline vs bearing wear and beat-up looseness.",
        source_type="synthetic",
    )

    # 3. Acoustic Feature Distributions & Statistical Tests
    rows = []
    for s in samples:
        f = extract_features_for_sample(s.audio, sample_rate=s.sample_rate)
        f["condition"] = s.metadata["condition"]
        rows.append(f)
    feat_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=feat_df, x="condition", y="kurtosis", ax=axes[0, 0], palette="Pastel1")
    axes[0, 0].set_title("Kurtosis (Impulsiveness)")
    
    sns.boxplot(data=feat_df, x="condition", y="spectral_centroid", ax=axes[0, 1], palette="Pastel1")
    axes[0, 1].set_title("Spectral Centroid (Hz) - Friction Brightness")

    sns.boxplot(data=feat_df, x="condition", y="high_to_low_energy_ratio", ax=axes[1, 0], palette="Pastel1")
    axes[1, 0].set_title("High-to-Low Energy Ratio")

    sns.boxplot(data=feat_df, x="condition", y="crest_factor", ax=axes[1, 1], palette="Pastel1")
    axes[1, 1].set_title("Crest Factor")
    plt.tight_layout()

    # Mann-Whitney U test on kurtosis between healthy and fault
    h_kurt = feat_df[feat_df["condition"] == "healthy"]["kurtosis"]
    f_kurt = feat_df[feat_df["condition"] == "fault"]["kurtosis"]
    u_stat, p_val = stats.mannwhitneyu(h_kurt, f_kurt)

    save_figure_with_sidecar(
        fig,
        fig_dir / "03_acoustic_feature_distributions.png",
        caption=f"Acoustic feature distributions by condition state. Mann-Whitney U test on Kurtosis p-value = {p_val:.4e}.",
        source_type="synthetic",
    )

    # 4. Cycle Boundaries Alignment
    cleaned = clean_audio(healthy_sample.audio, sample_rate=16000)
    cycles_res = segment_cycles(cleaned, sample_rate=16000)
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(cleaned)) / 16000
    ax.plot(t[:16000*3], cleaned[:16000*3], color="#2563EB", lw=1.0, label="Filtered Signal")
    for b in cycles_res.cycle_boundaries[:10]:
        if b / 16000 <= 3.0:
            ax.axvline(b / 16000, color="#DC2626", ls="--", lw=1.5, label="Cycle Boundary" if b == cycles_res.cycle_boundaries[0] else "")
    ax.set_title(f"Pick-Cycle Synchronous Boundary Detection (Est RPM: {cycles_res.estimated_rpm:.1f}, Conf: {cycles_res.confidence*100:.1f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_figure_with_sidecar(
        fig,
        fig_dir / "04_cycle_boundaries_alignment.png",
        caption="Waveform with detected pick-cycle boundaries anchored on recurring mechanical beat-up impacts.",
        source_type="synthetic",
    )

    # 5. Healthy Template & Residual Map
    template = build_healthy_template(cycles_res.normalized_cycles, sample_rate=16000)
    f_res = segment_cycles(clean_audio(bearing_sample.audio), sample_rate=16000)
    residual_map, _ = compute_cycle_residual(f_res.normalized_cycles[0], template, sample_rate=16000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im1 = axes[0].imshow(template["mean_spec"], origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_title("Healthy Baseline Pick-Cycle Template")
    axes[0].set_xlabel("Normalized Cycle Phase (%)")
    axes[0].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im1, ax=axes[0], label="Log Power")

    im2 = axes[1].imshow(residual_map, origin="lower", cmap="RdBu_r", aspect="auto")
    axes[1].set_title("Standardized Cycle Residual Heatmap (Bearing Wear)")
    axes[1].set_xlabel("Normalized Cycle Phase (%)")
    axes[1].set_ylabel("Mel Frequency Bin")
    plt.colorbar(im2, ax=axes[1], label="Z-Score Residual")
    plt.tight_layout()
    save_figure_with_sidecar(
        fig,
        fig_dir / "05_healthy_template_residual.png",
        caption="Learned healthy template vs cycle-synchronous residual heatmap highlighting localized high-frequency energy surge.",
        source_type="synthetic",
    )

    print(f"EDA complete! Publication figures written to {fig_dir}")


if __name__ == "__main__":
    main()
