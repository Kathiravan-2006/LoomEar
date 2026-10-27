# LoomEar Model Card

## 1. Model Overview
- **Model Name:** LoomEar Pick-Cycle-Synchronous Acoustic Ensemble
- **Version:** 1.0.0
- **Model Ladder:**
  - *Rung 1:* Heuristic Threshold (RMS + Kurtosis)
  - *Rung 2:* PCA Reconstruction Error, Isolation Forest, One-Class SVM
  - *Rung 3:* Random Forest & XGBoost Multi-class Fault Classifiers
  - *Rung 4:* 2D Spectrogram Convolutional Autoencoder

---

## 2. Intended Use
- **Primary Goal:** Unsupervised acoustic anomaly detection and early fault warning on power looms.
- **Input:** Single-channel airborne acoustic WAV (16 kHz, mono, 2 to 10 seconds).
- **Output:** Pick rate (RPM), continuous Anomaly Score, 0-100 Health Index, 3-State Category (`Healthy`, `Watch`, `Act`), and 2D Residual Error Heatmap with plain-English diagnosis.

---

## 3. Evaluation Summary (from `reports/metrics/final_results.json`)
- **Top Anomaly Detector (PCA Reconstruction):**
  - **ROC AUC:** 1.000 [95% CI: 1.000, 1.000]
  - **Partial AUC (FPR $\le$ 0.10):** 1.000
  - **Precision / Recall / F1:** 1.000 / 1.000 / 1.000
- **Zero-Shot Machine Transfer (LOMO on LOOM-04 Air-Jet):**
  - **ROC AUC:** 0.889 (strong zero-shot generalizability across differing loom mechanisms)
- **Multi-Class Classifier (XGBoost):**
  - **Macro F1:** 0.889
  - **Accuracy:** 0.889
- **Edge Inference Throughput:** 12.78 ms per 3-second audio window (>230x real-time).

---

## 4. Limitations
- Unsupervised anomaly detectors require 5–10 minutes of baseline healthy audio recorded during normal machine running to calibrate the healthy template.
- High external plant noise below 0 dB SNR degrades high-frequency sensitivity; close microphone mounting (30–50 cm) is mandatory.
