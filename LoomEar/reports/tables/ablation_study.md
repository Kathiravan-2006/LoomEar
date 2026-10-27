| Pipeline Configuration                            |   Test ROC AUC |   Delta vs Full |
|:--------------------------------------------------|---------------:|----------------:|
| Full Pipeline (Cycle-Sync + Residuals + Cleaning) |          0.942 |           0     |
| Ablation: Without Cycle-Synchronous Alignment     |          0.768 |          -0.174 |
| Ablation: Without Spectral Bandpass Filtering     |          0.815 |          -0.127 |
| Ablation: Without Residual Template Feature       |          0.834 |          -0.108 |
| Ablation: Without High-Frequency Bands            |          0.791 |          -0.151 |