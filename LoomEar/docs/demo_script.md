# LoomEar: 5-Minute Examination Demo Script

### Demo Objective
Demonstrate an end-to-end live diagnostic workflow: from acoustic capture, real-time pick-cycle segmentation, healthy template residual explanation, multi-loom fleet status, to ROI financial modeling.

---

### Step-by-Step Walkthrough

1. **Launch Dashboard:**
   - Command: `streamlit run app/dashboard.py`
   - Open browser at `http://localhost:8501`.

2. **Tab 1: Live Condition Monitor (Minute 1)**
   - Select **"Simulate Live Stream"** -> **"Normal Operational (Healthy)"**.
   - Show live waveform, pick-cycle boundary overlay, and `HEALTHY` green badge (`Health Index: ~24/100`).
   - Switch dropdown to **"Critical Bearing Failure (Act)"**.
   - Notice immediate shift of health gauge into red `ACT` zone (`Health Index: >85/100`), acoustic high-band hiss playback, and updated telemetry metrics (PPM: 212.0, Confidence: >85%).

3. **Tab 2: Cycle & Residual Explainability (Minute 2)**
   - Click on **"2. Cycle & Residual Explainability"**.
   - Show the **Healthy Baseline Template** vs **Standardized Residual Heatmap**.
   - Point to the bright red localized region in high frequencies (3.5 - 7.8 kHz) at mid-cycle beat-up.
   - Read the **Mechanic's Diagnostic Report**: *"Acoustic deviation concentrated in the High-Frequency band during the Mid-Pick Beat-Up phase."*

4. **Tab 3: Health Trend & Lead-Time (Minute 3)**
   - Click on **"3. Health Trend & Lead-Time"**.
   - Explain the 14-day progressive degradation trajectory.
   - Highlight the yellow **Early Warning Window** providing **3.5 days (84 hours)** of actionable lead time before machine seizure.

5. **Tab 4 & 5: Fleet Overview & Cost ROI Simulator (Minute 4 - 5)**
   - Review the plant-wide table in **Fleet Overview** showing LOOM-01 to LOOM-06 risk rankings.
   - Open **Savings & ROI Simulator**, drag the **"Number of Looms"** slider to 48 and **"Downtime Cost"** to $150/hr.
   - Highlight the live updated **$400,000+ Net Annual Savings** and sensitivity matrix.

---

### Offline Fallback Plan
If live Streamlit or audio output encounters any hardware issue:
- Pre-generated figures are available in `reports/figures/` (at 300 DPI).
- Standalone CLI demo runs via `python edge/record_and_score.py --max-cycles 5`.
- Verification metrics are frozen in `reports/metrics/final_results.json`.
