"""Audio processing, pick-cycle synchronous segmentation, and augmentation."""

from .preprocess import clean_audio, bandpass_filter, remove_dc, reduce_noise
from .cycles import (
    compute_envelope,
    estimate_pick_period,
    segment_cycles,
    time_normalize_cycle,
    build_healthy_template,
    compute_cycle_residual,
    CycleExtractionResult,
    evaluate_cycle_alignment_gain,
)
from .augment import spec_augment, add_noise_snr

__all__ = [
    "clean_audio",
    "bandpass_filter",
    "remove_dc",
    "reduce_noise",
    "compute_envelope",
    "estimate_pick_period",
    "segment_cycles",
    "time_normalize_cycle",
    "build_healthy_template",
    "compute_cycle_residual",
    "CycleExtractionResult",
    "evaluate_cycle_alignment_gain",
    "spec_augment",
    "add_noise_snr",
]
