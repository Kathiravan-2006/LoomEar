# LoomEar: Limitations and Future Roadmap

## 1. Current Scientific Limitations
1. **Single-Point Microphone Array:**
   - A single cardioid microphone relies on acoustic proximity (30–50 cm). Adjacent looms operating within 1 meter can introduce acoustic cross-talk if running at similar RPM without spatial beamforming.
2. **Template Calibration Requirement:**
   - Unsupervised anomaly detectors require ~5 minutes of baseline healthy audio recorded during standard machine operation to calibrate machine-specific baseline templates.
3. **Severe Ambient Noise Floor:**
   - While SNR robustness remains high down to 5 dB, extreme mill environments below 0 dB SNR diminish sensitivity to subtle incipient bearing micro-clicks.

---

## 2. Future Work & Research Extensions
1. **Multi-Microphone Spatial Beamforming:**
   - Deploying a dual-MEMS differential microphone array to dynamically steer acoustic nulls toward neighboring machines.
2. **Multi-Modal Sensor Fusion:**
   - Fusing airborne acoustic audio with single-phase motor drive current signatures (MCSA) for joint electrical-mechanical diagnostics.
3. **Edge Micro-Controller Quantization:**
   - Porting the ONNX model to Cortex-M4/M7 microcontrollers (e.g. STM32 / ESP32-S3) for sub-$15 all-in-one edge sensor pods.
