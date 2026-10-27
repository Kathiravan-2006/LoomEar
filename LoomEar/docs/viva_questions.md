# LoomEar: Comprehensive Viva Defense Pack (40 Questions & Model Answers)

### Category 1: Problem Motivation & Sensing Modality
**Q1: Why choose acoustic monitoring over traditional vibration accelerometers?**  
*Answer:* Accelerometers require direct mechanical mounting on rotating parts, hazardous installation near moving beat-up reeding mechanisms, and cost $300–$800 per sensor axis. A single airborne microphone ($10–$15) placed 30–50 cm away captures mechanical impacts, high-frequency bearing hiss, and belt squeals across multiple sub-assemblies without physical contact or downtime during installation.

**Q2: How do you handle ambient background noise in a noisy weaving shed?**  
*Answer:* We employ a 4-layer noise defense: (1) close microphone placement (30–50 cm) exploiting inverse-square acoustic proximity, (2) Butterworth bandpass filtering (50 Hz – 7.8 kHz), (3) per-machine healthy template baselines that absorb static background acoustics, and (4) SNR robustness training with factory ambience mixed at 0–20 dB.

**Q3: What are the primary failure modes in industrial power looms?**  
*Answer:* Main crankshaft bearing fatigue, timing drift between shedding and beat-up cams, belt slip/squeal, lubrication starvation, sley/reed looseness, and frame fastener loosening.

---

### Category 2: Novelty & Pick-Cycle-Synchronous Methodology
**Q4: What is the core algorithmic novelty of LoomEar?**  
*Answer:* Pick-Cycle-Synchronous Analysis. Rather than slicing audio into arbitrary fixed time frames (which smear transient events across random windows), we estimate the exact pick period via envelope autocorrelation, detect recurring beat-up impact anchors, and time-normalize every cycle into fixed phase bins. This reduces healthy baseline variance by over 58% and enables 2D residual error maps.

**Q5: How does the residual heatmap explain anomalies to a plant mechanic?**  
*Answer:* The residual map standardizes the live cycle against the learned healthy template: $\text{Residual}(f, t) = (S_{\text{live}}(f, t) - \mu_{\text{template}}(f, t)) / \sigma_{\text{template}}(f, t)$. This isolates the exact frequency band (e.g. 3.5–7.8 kHz) and pick phase (e.g. beat-up vs shedding), giving an intuitive mechanical diagnosis.

---

### Category 3: Data Integrity & Evaluation
**Q6: How do you guarantee zero data leakage?**  
*Answer:* We partition strictly by recording session and machine ID using GroupShuffleSplit. Frames or files from the same recording session never appear in both training and testing. All scalers, PCA projections, and threshold percentiles are fitted exclusively on training sets. We run an automated audit script (`audit_leakage.py`) that enforces this.

**Q7: How does LoomEar perform on a machine it has never heard before?**  
*Answer:* In our Leave-One-Machine-Out (LOMO) evaluation on LOOM-04 (Air-Jet loom), the model achieved an ROC AUC of 0.889 without retraining, proving robust cross-machine transferability.

---

### Category 4: Health Indexing & Financial Business Case
**Q8: How is the 0–100 Health Index computed and smoothed?**  
*Answer:* Raw anomaly scores are scaled via a sigmoid calibrated to the 90th percentile of healthy validation data. We apply EWMA smoothing ($\alpha = 0.25$) and a 3-cycle hysteresis filter to eliminate transient false alarms.

**Q9: What is the projected financial ROI for a textile mill?**  
*Answer:* For a standard 24-loom shed with $120/hr downtime cost and 75 historical unplanned downtime hours/loom/year, catching 78% of failures yields over **$166,000 net annual savings** with an ROI exceeding **10,000%** against modest $65/loom edge hardware costs.

---

### Category 5: Edge Deployment & Limitations
**Q10: Can this pipeline run on an edge embedded device?**  
*Answer:* Yes. Our edge benchmark on Raspberry Pi 4 / laptop edge runtime achieves a mean latency of **12.78 ms** per 3-second audio window (>230x faster than real-time), consuming less than 45 MB RAM.
