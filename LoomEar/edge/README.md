# LoomEar Edge Deployment Guide

This folder contains the standalone edge runtime scripts for deploying LoomEar acoustic condition monitoring on low-cost single-board computers (e.g. Raspberry Pi 4 Model B, Raspberry Pi Zero 2W, or NVIDIA Jetson Nano).

---

## 1. Hardware Architecture & Bill of Materials (BOM)

| Component | Specification | Estimated Cost (USD) |
| :--- | :--- | :--- |
| **SBC Compute Unit** | Raspberry Pi 4 (2GB RAM) or Pi Zero 2W | $35 - $45 |
| **Acoustic Sensor** | I2S MEMS Microphone (INMP441 / SPH0645) or USB Cardioid Mic | $8 - $15 |
| **Enclosure** | IP54 Dust/Lint-proof ventilated DIN-rail case | $8 |
| **Power Supply** | Industrial 24V DC to 5V 3A Step-Down Buck Converter | $6 |
| **Mounting Fixture** | Magnetic base articulated arm (30–50 cm from beat-up reed) | $10 |
| **Total Hardware BOM** | **Per Loom Edge Unit** | **~$67 - $84** |

---

## 2. Sensor Placement Guidelines

1. **Distance:** 30 cm to 50 cm from the loom's beat-up zone and main drive linkage.
2. **Orientation:** Cardioid capsule angled 45° toward the reed and main crank bearing, shielded from direct HVAC airflow.
3. **Vibration Isolation:** Use rubber grommets or magnetic clamp mounts to avoid direct structure-borne mechanical clipping.

---

## 3. Quickstart Edge Execution

```bash
# 1. Run local benchmark to test throughput
python edge/benchmark.py

# 2. Run real-time edge capture and telemetry logging
python edge/record_and_score.py --machine-id LOOM-01 --max-cycles 20

# 3. Export optimized edge models
python edge/export_onnx.py
```
