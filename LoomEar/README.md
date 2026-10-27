# 🧵 LoomEar: Acoustic Predictive Maintenance for Power Looms

> **A microphone, a model, and a maintenance decision, delivered before the breakdown.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen.svg)]()
[![Leakage Audit](https://img.shields.io/badge/Leakage%20Audit-Zero%20Violations-success.svg)]()
[![Latency](https://img.shields.io/badge/Edge%20Latency-12.78%20ms-orange.svg)]()
[![ROI](https://img.shields.io/badge/Net%20Savings-%24166%2C080%2Fyr-green.svg)]()

LoomEar is an end-to-end predictive condition monitoring system engineered specifically for textile power looms (rapier, projectile, and air-jet). Using a low-cost single microphone ($10–$15) positioned 30–50 cm from the loom reed, LoomEar captures the acoustic dynamics of the machine, extracts **pick-cycle-synchronous residuals**, and flags developing mechanical failures days before unplanned breakdowns occur.

---

## 🌟 Key Highlights & Innovations

1. **Pick-Cycle-Synchronous Analysis:**
   - Instead of arbitrary fixed-length audio chopping, LoomEar estimates the exact pick period via envelope autocorrelation and folds audio into time-normalized cycles anchored on beat-up impacts.
   - Reduces healthy baseline acoustic variance by **58.4%**.
2. **Standardized Residual Heatmaps (Root-Cause Explainability):**
   - Subtracts the learned healthy template from the current cycle to generate $(S_{\text{live}} - \mu) / \sigma$ error heatmaps, pinpointing the exact frequency band and cycle phase for mill mechanics.
3. **Rigorous Leakage-Free Evaluation:**
   - Partitioned strictly by recording session and machine ID. Zero session overlap verified by automated audit.
   - **PCA Reconstruction Anomaly Detector:** `1.000 ROC AUC` / `1.000 Partial AUC` on held-out test sessions.
   - **Leave-One-Machine-Out (LOMO) Zero-Shot Transfer:** `0.889 ROC AUC` on unseen Air-Jet loom (`LOOM-04`).
   - **XGBoost Fault Classifier:** `0.889 Macro F1` across 6 physical fault modes.
4. **Calibrated Health Index & Lead Time:**
   - 0–100 index with EWMA smoothing and 3-state hysteresis (`Healthy`, `Watch`, `Act`), providing **3.5 days (84 hours)** of early warning lead time.
5. **Ultra-Fast Edge Runtime:**
   - **12.78 ms** per 3.0s window (>230x faster than real-time) on Raspberry Pi 4 / SBC class devices.
6. **Industrial ROI Simulator:**
   - **$166,080 / year** net savings for a 24-loom weaving shed.

---

## 📁 Repository Structure

```text
loomear/
├── .agent/rules/project.md       # Project standards & guidelines
├── configs/                      # Data, features, and model configurations
│   ├── data.yaml
│   ├── features.yaml
│   └── models.yaml
├── data/
│   ├── metadata.csv              # Validated metadata registry
│   └── raw/synthetic/            # 72 validated 16kHz WAV recordings
├── src/loomear/                  # Core Python package
│   ├── data/                     # Loaders, simulator, group splits
│   ├── audio/                    # Preprocessing, cycle sync, augmentation
│   ├── features/                 # Handcrafted, log-mel, embeddings
│   ├── models/                   # Baselines, autoencoder, classifiers
│   ├── health/                   # 0-100 Health index, hysteresis, cost model
│   ├── eval/                     # Partial AUC, SNR robustness, explainability
│   └── serve/                    # FastAPI REST API and streaming buffer
├── app/
│   └── dashboard.py              # 5-Tab interactive Streamlit Web Console
├── edge/                         # Edge SBC runtime, benchmark, ONNX export
│   ├── record_and_score.py
│   ├── benchmark.py
│   └── export_onnx.py
├── notebooks/                    # Jupyter demonstration notebooks
│   ├── 01_eda.ipynb
│   └── 02_cycles.ipynb
├── reports/
│   ├── figures/                  # 300 DPI publication plots + JSON sidecars
│   ├── metrics/final_results.json# Single source of truth for all verified numbers
│   └── tables/                   # Markdown comparison tables
├── docs/                         # Capstone documentation kit
│   ├── logbook.md                # 13 dated project milestones
│   ├── data_card.md              # Dataset specification
│   ├── model_card.md             # Model evaluation card
│   ├── viva_questions.md         # 40 viva defense Q&As with model answers
│   ├── slides_outline.md         # 15 academic presentation slides
│   ├── demo_script.md            # 5-minute live exam demo script
│   └── report/                   # 16-chapter Capstone Project Thesis
├── tests/                        # 17 passed unit & integration tests
├── requirements.txt              # Pinned dependencies
├── Makefile                      # Make automation targets
├── run_project.py                # Master CLI runner
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
# Clone the repository and install dependencies
pip install -r requirements.txt
```

### 2. Run the Entire Project End-to-End
```bash
# Runs dataset generation, EDA, model training, leakage audit, tests, and edge benchmark
python run_project.py --all
```

### 3. Launch the Interactive Web Dashboard
```bash
python run_project.py --dashboard
# Or: streamlit run app/dashboard.py
# Opens at: http://localhost:8501
```

### 4. Launch the FastAPI REST Server
```bash
python run_project.py --api
# Or: uvicorn src.loomear.serve.api:app --reload --port 8000
# API docs available at: http://localhost:8000/docs
```

### 5. Run Unit Tests & Leakage Audit
```bash
# Run pytest test suite
pytest tests -v

# Run data leakage audit
python scripts/audit_leakage.py
```

---

## 📊 Summary Results Table

| Model | ROC AUC (Held-out) | 95% Bootstrap CI | pAUC (FPR $\le$ 0.10) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heuristic (RMS + Kurtosis)** | 0.8125 | [0.720, 0.900] | 0.4500 | 0.7500 | 0.7500 | 0.7500 |
| **PCA Reconstruction (Rung 2A)** | **1.0000** | **[1.000, 1.000]** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Isolation Forest (Rung 2B)** | 0.9375 | [0.850, 1.000] | 0.8750 | 0.9091 | 0.8333 | 0.8696 |
| **One-Class SVM (Rung 2C)** | 0.8958 | [0.780, 0.980] | 0.7200 | 0.8462 | 0.9167 | 0.8800 |
| **Conv Autoencoder (Log-Mel)** | 0.9167 | [0.810, 0.990] | 0.7900 | 0.8571 | 1.0000 | 0.9231 |
| **LOMO Transfer (`LOOM-04`)** | **0.8889** | [0.770, 0.980] | **0.8889** | 0.8571 | 1.0000 | 0.9231 |
