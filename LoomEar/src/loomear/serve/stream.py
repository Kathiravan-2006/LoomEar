"""
Real-time streaming audio buffer processor for continuous loom condition monitoring.
"""

from typing import Callable, Optional
import numpy as np


class AudioStreamProcessor:
    """
    Manages a circular sliding buffer of live audio samples and invokes
    the inference pipeline on fixed duration hops.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration_sec: float = 2.0,
        hop_duration_sec: float = 1.0,
        callback: Optional[Callable] = None,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(window_duration_sec * sample_rate)
        self.hop_samples = int(hop_duration_sec * sample_rate)
        self.callback = callback
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.samples_since_last_hop = 0

    def push(self, new_chunk: np.ndarray):
        """Pushes incoming raw audio chunk into sliding buffer."""
        chunk_len = len(new_chunk)
        if chunk_len == 0:
            return

        # Shift buffer and insert new chunk
        if chunk_len >= self.window_samples:
            self.buffer = new_chunk[-self.window_samples:].astype(np.float32)
            self.samples_since_last_hop = self.hop_samples
        else:
            self.buffer[:-chunk_len] = self.buffer[chunk_len:]
            self.buffer[-chunk_len:] = new_chunk.astype(np.float32)
            self.samples_since_last_hop += chunk_len

        # Check if hop interval reached
        if self.samples_since_last_hop >= self.hop_samples:
            self.samples_since_last_hop = 0
            if self.callback:
                self.callback(self.buffer.copy())

    def get_current_window(self) -> np.ndarray:
        """Returns the current window buffer."""
        return self.buffer.copy()
