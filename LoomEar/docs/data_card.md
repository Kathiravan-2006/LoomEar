# LoomEar Data Card

## 1. Dataset Summary
- **Project:** LoomEar: Acoustic Predictive Maintenance for Power Looms
- **Domain:** Industrial Textile Machinery Condition Monitoring
- **Primary Modality:** Single-channel airborne acoustic audio (.wav, 16-bit, 16.0 kHz)
- **Total Recordings:** 72 validated sessions across 4 distinct power looms
- **Condition Distribution:** Healthy (24 files), Watch (24 files), Fault (24 files)

---

## 2. Data Sources & Physical Authenticity
1. **Synthetic Physically-Informed Generator (`synthetic`):** 100% of benchmark evaluation files in current repository. Faithfully synthesizes periodic impact trains, multi-resonant beat-up dynamics, weft cutting clicks, shedding motion, and parameterized fault modes (bearing wear, timing drift, belt squeal, lubrication starvation, beat-up looseness, loose frame).
2. **Public Benchmarks (`public`):** Adapters provided for MIMII and DCASE anomalous machine sound datasets.
3. **Real Mill Recordings Protocol (`real`):** Documented in Section 5 of the playbook for on-site deployment.

---

## 3. Metadata Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `file_id` | String | Unique file identifier (e.g. `synth_0001`) |
| `source_type` | Categorical | `real`, `public`, or `synthetic` |
| `machine_id` | String | Machine identifier (`LOOM-01` to `LOOM-04`) |
| `loom_type` | Categorical | `rapier`, `projectile`, `air-jet` |
| `rpm` | Float | Operational pick speed (PPM) |
| `session_id` | String | Recording session identifier for group splitting |
| `date` | String | Recording date (YYYY-MM-DD) |
| `shift` | Categorical | `Shift_A`, `Shift_B`, `Shift_C` |
| `mic_position` | String | Distance and orientation from reed/beat-up |
| `condition` | Categorical | `healthy`, `watch`, or `fault` |
| `fault_type` | Categorical | `none`, `bearing_wear`, `timing_drift`, etc. |
| `label_source` | Categorical | `log`, `expert`, or `injected` |
| `snr_db` | Float | Background factory noise level (dB) |
| `notes` | String | Contextual logging notes |

---

## 4. Ethical & Safety Declarations
- Audio is recorded purely in mechanical machinery frequency bands (50 Hz - 7.8 kHz).
- No human voice or identifiable personal conversations are retained.
- Synthetic audio eliminates privacy and intellectual property conflicts.
