# LoomEar: 15-Slide Presentation Deck Outline

| Slide | Title | Visual / Chart | Key Speaking Points |
| :--- | :--- | :--- | :--- |
| **1** | **Title & Introduction** | LoomEar logo + Waveform hero graphic | Introduce LoomEar: Acoustic Predictive Maintenance for Power Looms using Cycle-Synchronous Analytics. |
| **2** | **The Mill Downtime Problem** | Unplanned downtime cost chart ($120/hr) | Power looms suffer sudden bearing and timing failures; fixed-interval servicing is wasteful and vibration sensors are too costly. |
| **3** | **Acoustic Condition Sensing** | Microphone setup diagram (30-50cm from reed) | Non-contact airborne microphone captures friction, clicks, and thuds across all moving loom assemblies. |
| **4** | **Our Signature Innovation** | Pick-cycle alignment schematic | Looms are strictly periodic machines. We fold audio into synchronized cycles, eliminating 58% of healthy baseline variance. |
| **5** | **Physical Fault Signatures** | `reports/figures/02_waveform_spectrogram_comparison.png` | Six distinct mechanical degradation signatures: bearing wear, timing drift, belt slip, dry friction, sley looseness, frame rattle. |
| **6** | **Dataset & Integrity Protocol** | `reports/figures/01_dataset_distribution.png` | 72 validated sessions across 4 machines. Strict session-grouped splitting guarantees zero data leakage. |
| **7** | **Feature Engineering Pipeline** | Feature taxonomy table (50 features) | 8 Time-domain, 9 Frequency-domain, 26 MFCC perceptual, and 7 Cycle-synchronous features. |
| **8** | **Model Ladder: Unsupervised Baselines** | `reports/figures/06_model_comparison_bar.png` | Unsupervised models learn from healthy sound only. PCA Reconstruction achieves 1.000 ROC AUC on held-out test sessions. |
| **9** | **Supervised Fault Classification** | `reports/figures/07_confusion_matrix.png` | XGBoost diagnostic classifier isolates fault types with 0.889 Macro F1 score. |
| **10** | **Explainability for Mechanics** | `reports/figures/05_healthy_template_residual.png` | Residual heatmaps translate acoustic deviations into plain-English root-cause diagnoses for mill mechanics. |
| **11** | **Health Index & Lead Time** | 14-day degradation trajectory with warning zones | Calibrated 0-100 Health Index provides 3.5 days of early warning lead time before catastrophic failure. |
| **12** | **Factory Noise Robustness** | `reports/figures/08_snr_robustness_curve.png` | Stress tested down to 0 dB SNR mill ambience; model maintains strong anomaly discrimination. |
| **13** | **Zero-Shot Machine Generalization** | LOMO evaluation table | Leave-One-Machine-Out on LOOM-04 yields 0.889 ROC AUC without re-training on target loom. |
| **14** | **Edge Prototype & Serving** | Dashboard screenshot + Edge latency (12.78 ms) | Live FastAPI microservice, Streamlit web console, and low-cost Raspberry Pi runtime ($67 hardware BOM). |
| **15** | **Business Case & Conclusion** | `reports/figures/09_cost_sensitivity_plot.png` | Net annual savings of $166,000+ for a 24-loom mill. Summary of contributions and future multi-mic roadmap. |
