"""Data management, loading, synthetic generation and splitting for LoomEar."""

from .loaders import load_audio, load_metadata, validate_metadata, AudioSample
from .simulator import SyntheticLoomSimulator, generate_dataset
from .splits import get_session_split, get_lomo_split, audit_data_leakage

__all__ = [
    "load_audio",
    "load_metadata",
    "validate_metadata",
    "AudioSample",
    "SyntheticLoomSimulator",
    "generate_dataset",
    "get_session_split",
    "get_lomo_split",
    "audit_data_leakage",
]
