"""
Physics-informed synthetic power loom audio simulator with fault injection.
Accurately models pick cycle mechanics and realistic acoustic degradation modes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf

logger = logging.getLogger(__name__)


class SyntheticLoomSimulator:
    """
    Simulates acoustic signals of weaving power looms under healthy
    and anomalous conditions with pick-cycle synchronicity.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        default_rpm: float = 210.0,
        random_seed: Optional[int] = 42,
    ):
        self.sample_rate = sample_rate
        self.default_rpm = default_rpm
        self.rng = np.random.default_rng(random_seed)

    def generate_ambience(self, duration_sec: float) -> np.ndarray:
        """
        Synthesizes a realistic textile mill ambient background:
        low-frequency HVAC hum, distant machinery rumble, and diffuse pink noise.
        """
        n_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # 50 Hz electrical hum + 100/150 Hz harmonics
        mains_hum = 0.04 * np.sin(2 * np.pi * 50 * t) + 0.015 * np.sin(2 * np.pi * 100 * t)

        # Pink / 1/f noise for diffuse mill ambience
        white = self.rng.standard_normal(n_samples)
        b, a = signal.butter(2, [30.0 / (self.sample_rate / 2), 2500.0 / (self.sample_rate / 2)], btype="band")
        diffuse = signal.lfilter(b, a, white) * 0.06

        ambience = mains_hum + diffuse
        return ambience.astype(np.float32)

    def _generate_impact(
        self,
        duration_samples: int,
        primary_freq: float = 280.0,
        decay_rate: float = 45.0,
        harmonics: Tuple[float, ...] = (1.0, 2.2, 4.8, 8.5),
        weights: Tuple[float, ...] = (1.0, 0.6, 0.35, 0.15),
    ) -> np.ndarray:
        """Generates a multi-resonant damped mechanical impact impulse."""
        t = np.arange(duration_samples) / self.sample_rate
        envelope = np.exp(-decay_rate * t)
        impulse = np.zeros(duration_samples)
        for h, w in zip(harmonics, weights):
            freq = primary_freq * h
            if freq < self.sample_rate / 2:
                impulse += w * np.sin(2 * np.pi * freq * t + self.rng.uniform(0, 2 * np.pi))
        return envelope * impulse

    def generate_healthy_loom(
        self,
        duration_sec: float = 10.0,
        rpm: Optional[float] = None,
        jitter_std: float = 0.008,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Generates healthy baseline loom audio comprising periodic pick cycles:
        - Shedding motion (smooth friction)
        - Beat-up impact (dominant impulse)
        - Weft insertion and rapier/shuttle return
        """
        if rpm is None:
            rpm = self.default_rpm

        pick_period = 60.0 / rpm
        n_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        cycle_starts = []

        current_time = self.rng.uniform(0, pick_period * 0.5)
        while current_time < duration_sec:
            cycle_starts.append(current_time)
            period_jitter = pick_period * (1.0 + self.rng.normal(0, jitter_std))
            cycle_samples = int(period_jitter * self.sample_rate)
            start_idx = int(current_time * self.sample_rate)
            end_idx = min(start_idx + cycle_samples, n_samples)
            actual_len = end_idx - start_idx

            if actual_len <= 0:
                break

            # 1. Beat-up event (at ~ 30% of pick cycle)
            beatup_pos = int(0.30 * actual_len)
            beatup_len = min(int(0.12 * self.sample_rate), actual_len - beatup_pos)
            if beatup_len > 0:
                impulse = self._generate_impact(beatup_len, primary_freq=320.0, decay_rate=40.0)
                amp_var = self.rng.uniform(0.92, 1.08)
                audio[start_idx + beatup_pos : start_idx + beatup_pos + beatup_len] += amp_var * impulse[:beatup_len]

            # 2. Shedding and rapier travel (continuous texture across cycle)
            t_cycle = np.arange(actual_len) / self.sample_rate
            shed_noise = self.rng.standard_normal(actual_len)
            b, a = signal.butter(2, [300.0 / (self.sample_rate / 2), 2200.0 / (self.sample_rate / 2)], btype="band")
            shed_texture = signal.lfilter(b, a, shed_noise) * 0.08
            shed_mod = (0.5 + 0.5 * np.sin(2 * np.pi * (1.0 / period_jitter) * t_cycle)) ** 2
            audio[start_idx:end_idx] += shed_texture * shed_mod

            # 3. Weft cutter click (at ~ 85% of cycle)
            click_pos = int(0.85 * actual_len)
            click_len = min(int(0.02 * self.sample_rate), actual_len - click_pos)
            if click_len > 0:
                click_impulse = self._generate_impact(click_len, primary_freq=1800.0, decay_rate=160.0)
                audio[start_idx + click_pos : start_idx + click_pos + click_len] += 0.35 * click_impulse[:click_len]

            # 4. Motor shaft baseline hum (rpm harmonic)
            motor_f = rpm / 60.0 * 4.0
            audio[start_idx:end_idx] += 0.05 * np.sin(2 * np.pi * motor_f * t_cycle)

            current_time += period_jitter

        return audio, cycle_starts

    def inject_fault(
        self,
        clean_audio: np.ndarray,
        fault_type: str,
        severity: float = 0.5,
        rpm: float = 210.0,
    ) -> np.ndarray:
        """
        Injects specific physical degradation acoustics.
        Severity ranges from 0.0 (no fault) to 1.0 (extreme failure).
        """
        audio = clean_audio.copy()
        n_samples = len(audio)
        t = np.arange(n_samples) / self.sample_rate
        pick_period = 60.0 / rpm

        if fault_type == "none" or severity <= 0.0:
            return audio

        if fault_type == "bearing_wear":
            # High frequency broadband friction (3.5 - 7.5 kHz) + sharp micro-clicks
            noise = self.rng.standard_normal(n_samples)
            b, a = signal.butter(3, [3500.0 / (self.sample_rate / 2), 7600.0 / (self.sample_rate / 2)], btype="band")
            hf_hiss = signal.lfilter(b, a, noise)
            
            # Bearing defect ball-pass frequency clicks (approx 6.8x shaft speed)
            bpfi = (rpm / 60.0) * 6.8
            click_train = np.zeros(n_samples)
            click_interval = int(self.sample_rate / bpfi)
            for idx in range(0, n_samples, click_interval):
                c_len = min(int(0.003 * self.sample_rate), n_samples - idx)
                if c_len > 0:
                    click_train[idx : idx + c_len] += np.exp(-300.0 * np.arange(c_len) / self.sample_rate)

            fault_signal = (0.7 * hf_hiss + 0.5 * click_train) * severity * 0.85
            audio += fault_signal

        elif fault_type == "timing_drift":
            # Mechanical timing shift: beat-up impact happens prematurely or with erratic phase drift
            phase_warp = severity * 0.04 * np.sin(2 * np.pi * (1.0 / (pick_period * 4.0)) * t)
            warp_indices = np.clip(
                np.arange(n_samples) + (phase_warp * self.sample_rate).astype(int),
                0,
                n_samples - 1,
            )
            # Add secondary out-of-phase impact shudder
            shudder = np.roll(audio, int(0.035 * self.sample_rate)) * (0.35 * severity)
            audio = audio[warp_indices] + shudder

        elif fault_type == "belt_squeal":
            # Narrowband high-pitch friction resonance with slip amplitude modulation
            squeal_f = 1650.0 + 200.0 * np.sin(2 * np.pi * 3.5 * t)
            phase = 2 * np.pi * np.cumsum(squeal_f) / self.sample_rate
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * (rpm / 60.0 * 2.0) * t)
            squeal = np.sin(phase) * mod * (severity * 0.65)
            audio += squeal

        elif fault_type == "lubrication_loss":
            # Severe metal-to-metal dry contact: boosts high frequencies and spectral centroid
            b, a = signal.butter(2, 2000.0 / (self.sample_rate / 2), btype="high")
            hf_dry = signal.lfilter(b, a, audio)
            audio += hf_dry * (severity * 1.2)

        elif fault_type == "beatup_looseness":
            # Sley / reed mount looseness: heavier, doubled impact with exaggerated sub-harmonic bass ring
            b, a = signal.butter(2, [60.0 / (self.sample_rate / 2), 350.0 / (self.sample_rate / 2)], btype="band")
            heavy_thud = signal.lfilter(b, a, audio) * (1.5 * severity)
            secondary_echo = np.roll(audio, int(0.045 * self.sample_rate)) * (0.45 * severity)
            audio += heavy_thud + secondary_echo

        elif fault_type == "loose_frame":
            # Random metallic rattles and transients occurring asynchronously between pick cycles
            rattle_noise = self.rng.standard_normal(n_samples)
            b, a = signal.butter(2, [1200.0 / (self.sample_rate / 2), 4800.0 / (self.sample_rate / 2)], btype="band")
            rattle = signal.lfilter(b, a, rattle_noise)
            # Burst gating
            gate = (self.rng.uniform(0, 1, n_samples) > (1.0 - 0.05 * severity)).astype(float)
            smooth_gate = np.convolve(gate, np.hanning(int(0.01 * self.sample_rate)), mode="same")
            audio += rattle * smooth_gate * (0.6 * severity)

        else:
            logger.warning(f"Unknown fault type: {fault_type}, returning unaltered signal.")

        return audio

    def mix_ambience_snr(
        self,
        signal_audio: np.ndarray,
        ambience_audio: Optional[np.ndarray] = None,
        snr_db: float = 15.0,
    ) -> np.ndarray:
        """
        Mixes signal with factory ambient noise at a calibrated Signal-to-Noise Ratio (dB).
        """
        n_samples = len(signal_audio)
        if ambience_audio is None or len(ambience_audio) < n_samples:
            ambience = self.generate_ambience(duration_sec=n_samples / self.sample_rate)
        else:
            ambience = ambience_audio[:n_samples]

        sig_power = np.mean(signal_audio ** 2) + 1e-12
        amb_power = np.mean(ambience ** 2) + 1e-12

        # SNR_db = 10 * log10(P_sig / P_noise) => P_noise_target = P_sig / (10^(SNR/10))
        target_amb_power = sig_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_amb_power / amb_power)

        mixed = signal_audio + scale * ambience
        # Peak normalise to prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 0.98:
            mixed = mixed / peak * 0.95

        return mixed.astype(np.float32)


def generate_dataset(
    output_dir: Union[str, Path],
    num_sessions: int = 16,
    duration_per_file_sec: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a full synthetic benchmark dataset with realistic sessions,
    healthy baselines, multiple loom machines, and distinct physical faults at varying SNRs.
    """
    out_path = Path(output_dir)
    audio_dir = out_path / "raw" / "synthetic"
    audio_dir.mkdir(parents=True, exist_ok=True)

    simulator = SyntheticLoomSimulator(sample_rate=16000, random_seed=seed)
    rng = np.random.default_rng(seed)

    machines = [
        {"id": "LOOM-01", "type": "rapier", "rpm": 210.0, "mic": "35cm_drive_side"},
        {"id": "LOOM-02", "type": "rapier", "rpm": 215.0, "mic": "40cm_drive_side"},
        {"id": "LOOM-03", "type": "projectile", "rpm": 195.0, "mic": "45cm_weft_side"},
        {"id": "LOOM-04", "type": "air-jet", "rpm": 240.0, "mic": "30cm_front_sley"},
        {"id": "LOOM-05", "type": "rapier", "rpm": 210.0, "mic": "35cm_drive_side"},
        {"id": "LOOM-06", "type": "rapier", "rpm": 208.0, "mic": "38cm_drive_side"},
        {"id": "LOOM-07", "type": "air-jet", "rpm": 245.0, "mic": "30cm_front_sley"},
        {"id": "LOOM-08", "type": "projectile", "rpm": 190.0, "mic": "45cm_weft_side"},
    ]

    fault_types = [
        "bearing_wear",
        "timing_drift",
        "belt_squeal",
        "lubrication_loss",
        "beatup_looseness",
        "loose_frame",
    ]

    records = []
    file_counter = 1

    # Pre-generate ambient background clip for noise analysis and mixing
    ambience_clip = simulator.generate_ambience(duration_sec=30.0)
    amb_path = audio_dir / "ambience_loom_off.wav"
    sf.write(str(amb_path), ambience_clip, 16000)

    # 1. Generate Healthy Baselines for every machine across multiple shifts/sessions
    for m in machines:
        for s in range(1, 4):  # 3 healthy sessions per machine
            session_id = f"SESS-{m['id']}-H{s:02d}"
            date_str = f"2026-03-{10 + s:02d}"
            shift = "Shift_A" if s % 2 == 1 else "Shift_B"

            for rep in range(2):
                file_id = f"synth_{file_counter:04d}"
                file_counter += 1

                clean, _ = simulator.generate_healthy_loom(
                    duration_sec=duration_per_file_sec,
                    rpm=m["rpm"] + rng.uniform(-1.5, 1.5),
                )
                snr = rng.choice([20.0, 15.0, 12.0])
                audio = simulator.mix_ambience_snr(clean, ambience_clip, snr_db=snr)

                file_path = audio_dir / f"{file_id}.wav"
                sf.write(str(file_path), audio, 16000)

                records.append({
                    "file_id": file_id,
                    "source_type": "synthetic",
                    "machine_id": m["id"],
                    "loom_type": m["type"],
                    "rpm": m["rpm"],
                    "session_id": session_id,
                    "date": date_str,
                    "shift": shift,
                    "mic_position": m["mic"],
                    "condition": "healthy",
                    "fault_type": "none",
                    "label_source": "injected",
                    "snr_db": float(snr),
                    "notes": "Healthy operational baseline recording",
                })

    # 2. Generate Fault and Watch conditions with progressive severities
    for m in machines:
        for f_type in fault_types:
            session_id = f"SESS-{m['id']}-{f_type[:4].upper()}"
            date_str = "2026-03-18"
            shift = "Shift_C"

            # Watch condition (moderate degradation)
            file_id = f"synth_{file_counter:04d}"
            file_counter += 1
            clean, _ = simulator.generate_healthy_loom(duration_sec=duration_per_file_sec, rpm=m["rpm"])
            faulty = simulator.inject_fault(clean, fault_type=f_type, severity=0.35, rpm=m["rpm"])
            audio = simulator.mix_ambience_snr(faulty, ambience_clip, snr_db=15.0)
            sf.write(str(audio_dir / f"{file_id}.wav"), audio, 16000)

            records.append({
                "file_id": file_id,
                "source_type": "synthetic",
                "machine_id": m["id"],
                "loom_type": m["type"],
                "rpm": m["rpm"],
                "session_id": session_id,
                "date": date_str,
                "shift": shift,
                "mic_position": m["mic"],
                "condition": "watch",
                "fault_type": f_type,
                "label_source": "injected",
                "snr_db": 15.0,
                "notes": f"Early stage developing {f_type} (severity 0.35)",
            })

            # Fault condition (high severity)
            file_id = f"synth_{file_counter:04d}"
            file_counter += 1
            clean, _ = simulator.generate_healthy_loom(duration_sec=duration_per_file_sec, rpm=m["rpm"])
            faulty = simulator.inject_fault(clean, fault_type=f_type, severity=0.85, rpm=m["rpm"])
            audio = simulator.mix_ambience_snr(faulty, ambience_clip, snr_db=10.0)
            sf.write(str(audio_dir / f"{file_id}.wav"), audio, 16000)

            records.append({
                "file_id": file_id,
                "source_type": "synthetic",
                "machine_id": m["id"],
                "loom_type": m["type"],
                "rpm": m["rpm"],
                "session_id": session_id,
                "date": date_str,
                "shift": shift,
                "mic_position": m["mic"],
                "condition": "fault",
                "fault_type": f_type,
                "label_source": "injected",
                "snr_db": 10.0,
                "notes": f"Critical developed {f_type} (severity 0.85)",
            })

    metadata_df = pd.DataFrame(records)
    metadata_path = out_path / "metadata.csv"
    metadata_df.to_csv(str(metadata_path), index=False)
    logger.info(f"Generated {len(metadata_df)} synthetic recordings with metadata saved to {metadata_path}")
    return metadata_df
