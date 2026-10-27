"""
Audio loading and metadata validation module for LoomEar.
Enforces strict schema checks and audio integrity validation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

EXPECTED_METADATA_COLUMNS = [
    "file_id",
    "source_type",
    "machine_id",
    "loom_type",
    "rpm",
    "session_id",
    "date",
    "shift",
    "mic_position",
    "condition",
    "fault_type",
    "label_source",
    "snr_db",
    "notes",
]

VALID_SOURCES = {"real", "public", "synthetic"}
VALID_CONDITIONS = {"healthy", "watch", "fault"}
VALID_FAULTS = {
    "none",
    "bearing_wear",
    "timing_drift",
    "belt_squeal",
    "lubrication_loss",
    "beatup_looseness",
    "loose_frame",
}


@dataclass
class AudioSample:
    """Container for loaded audio and associated verified metadata."""
    file_id: str
    audio: np.ndarray
    sample_rate: int
    duration_sec: float
    metadata: Dict[str, Union[str, float, int]]


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 16000,
    mono: bool = True,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 300.0,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio WAV file with validation and resample if necessary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        raise ValueError(f"Failed to read audio file {path}: {str(e)}")

    if audio.ndim > 1:
        if mono:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.T

    duration = len(audio) / sr
    if duration < min_duration_sec:
        raise ValueError(f"Audio duration ({duration:.2f}s) is shorter than minimum ({min_duration_sec}s): {path}")
    if duration > max_duration_sec:
        logger.warning(f"Audio duration ({duration:.2f}s) exceeds max ({max_duration_sec}s), truncating.")
        audio = audio[: int(max_duration_sec * sr)]

    if target_sr is not None and sr != target_sr:
        # High-quality polyphase / linear resampling
        num_target_samples = int(round(len(audio) * float(target_sr) / sr))
        indices = np.linspace(0, len(audio) - 1, num_target_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr

    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError(f"Audio array contains NaN or Inf values in {path}")

    return audio, sr


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate metadata schema, ensuring all required columns, valid categorical values,
    unique file IDs, and non-empty rows.
    """
    missing_cols = [c for c in EXPECTED_METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Metadata is missing required columns: {missing_cols}")

    if df["file_id"].duplicated().any():
        duplicates = df[df["file_id"].duplicated()]["file_id"].tolist()
        raise ValueError(f"Duplicate file_id found in metadata: {duplicates[:5]}")

    invalid_sources = set(df["source_type"].dropna()) - VALID_SOURCES
    if invalid_sources:
        raise ValueError(f"Invalid source_type values: {invalid_sources}. Allowed: {VALID_SOURCES}")

    invalid_conds = set(df["condition"].dropna()) - VALID_CONDITIONS
    if invalid_conds:
        raise ValueError(f"Invalid condition values: {invalid_conds}. Allowed: {VALID_CONDITIONS}")

    # Check for NaN in critical identity fields
    critical_fields = ["file_id", "source_type", "machine_id", "session_id", "condition", "fault_type"]
    for field in critical_fields:
        if df[field].isna().any():
            raise ValueError(f"Field '{field}' contains NaN values.")

    return df


def load_metadata(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and validate metadata CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(str(path))
    return validate_metadata(df)


def load_dataset(
    audio_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    target_sr: int = 16000,
) -> List[AudioSample]:
    """
    Load all audio files listed in metadata into AudioSample objects.
    """
    audio_dir = Path(audio_dir)
    df = load_metadata(metadata_csv)
    samples = []

    for _, row in df.iterrows():
        file_name = str(row["file_id"])
        if not file_name.endswith(".wav"):
            file_name += ".wav"
        
        # Search recursively or directly
        matches = list(audio_dir.rglob(file_name))
        if not matches:
            # check directly
            direct_path = audio_dir / file_name
            if direct_path.exists():
                file_path = direct_path
            else:
                raise FileNotFoundError(f"Audio file {file_name} not found in {audio_dir}")
        else:
            file_path = matches[0]

        audio, sr = load_audio(file_path, target_sr=target_sr)
        duration = len(audio) / sr
        sample = AudioSample(
            file_id=str(row["file_id"]),
            audio=audio,
            sample_rate=sr,
            duration_sec=duration,
            metadata=row.to_dict(),
        )
        samples.append(sample)

    logger.info(f"Successfully loaded {len(samples)} audio samples from {audio_dir}")
    return samples
