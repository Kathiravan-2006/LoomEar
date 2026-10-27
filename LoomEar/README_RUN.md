# LoomEar: Windows Run and Verification Guide

## Project Overview

LoomEar is a Python prototype for acoustic condition monitoring of power looms. The checked-in benchmark is synthetic: 144 labeled WAV recordings plus one ambient-noise WAV. No real mill recordings, database, API credentials, or external service are required for the local demo.

The opened workspace is the parent folder. The actual project root, containing `requirements.txt` and `run_project.py`, is `LoomEar\` beneath it. Run project commands from that directory.

## Architecture and Data Flow

1. `data/metadata.csv` associates recording IDs with machine, session, condition, and source labels. `data/raw/synthetic/` contains the WAVs.
2. `src/loomear/data/loaders.py` validates metadata and loads/resamples recordings. `src/loomear/audio/preprocess.py` removes DC, filters, and normalizes audio; noise reduction is optional and disabled by default in `clean_audio`.
3. `src/loomear/audio/cycles.py` estimates pick period, segments cycles, builds log-mel healthy templates, and computes residual scores. `src/loomear/features/` contains handcrafted and spectrogram features.
4. `src/loomear/models/` contains heuristic, PCA, Isolation Forest, One-Class SVM, Random Forest, XGBoost, and a class named `ConvAutoencoderAnomalyDetector`. That last implementation is PCA reconstruction, not a neural/convolutional network.
5. `src/loomear/eval/` provides metrics, robustness, and explanation helpers; `src/loomear/health/` provides index, decision, and cost-model helpers.
6. `scripts/train_and_evaluate.py` builds evaluation results and writes reports. `scripts/audit_leakage.py` checks session splits, the separate leave-one-machine-out split, and source labels.
7. `app/dashboard.py` is a six-view Streamlit demo. `src/loomear/serve/api.py` is a FastAPI demo. `edge/` contains a synthetic monitor, timing benchmark, and manifest writer.

The three `configs/*.yaml` files document settings, but the Python scripts use many hard-coded values and do not load these YAML files. Do not assume changing a YAML value changes a run.

## Prerequisites

- Windows 11 and standard CPython 3.11 or newer. The project README declares Python 3.11+.
- VS Code with the Python extension. Install the Jupyter extension to run the notebooks.
- No database, credentials, microphone, or network service is needed for the simulated demo.

In the inspected shell, `py` was unavailable. `python.exe` resolved to `C:\Espressif\tools\python\python.exe` (Python 3.11.15). A venv created under `%TEMP%` worked, but creating `.venv\Scripts\python.exe` inside the extracted project failed with `Permission denied`. Use a venv outside the extracted folder. An incomplete ignored `.venv` folder was left by that attempt; these instructions intentionally do not delete it.

## Setup in PowerShell

Open PowerShell in the workspace parent, then run:

```powershell
Set-Location .\LoomEar
$Python = (Get-Command python.exe).Source
& $Python --version
$Venv = Join-Path $env:TEMP 'loomear-verify-venv'
& $Python -m venv $Venv
& "$Venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

The commands above use the Python executable available in the current Windows shell. In this inspection that was Espressif's Python 3.11.15; creating the environment under `%TEMP%` avoided the permission error seen under `Downloads`. A normal CPython 3.11+ installation also works. If no suitable `python.exe` is found, install CPython 3.11+ and reopen PowerShell. If PowerShell blocks activation for this window, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

VS Code: open the actual `LoomEar` project folder, then select the interpreter at `$Venv\Scripts\python.exe` using **Python: Select Interpreter**. For notebook execution, choose the same environment in the notebook kernel picker. If VS Code does not offer it, install `ipykernel` into this environment with `python -m pip install ipykernel`; this is only needed for notebook kernels and is not an application dependency.

`requirements.txt` contains minimum-version constraints rather than fully pinned versions. It now includes direct imports missing from the original manifest: Librosa (handcrafted features), Plotly (dashboard), `python-multipart` (FastAPI file/form requests), and Tabulate (pandas Markdown tables). SHAP is imported lazily only by the optional tree explainer and is not needed for the API or tests.

## Safe First Checks

Run these before any pipeline that regenerates files:

```powershell
python -m pytest tests -v
python scripts/audit_leakage.py
```

The leakage audit reads the existing metadata when it exists. Both commands passed in the temporary Python 3.11.15 environment during this inspection.

## Run the Application and Features

Run from the project root with the virtual environment active:

| Feature | Command | Expected result and output |
|---|---|---|
| Master runner help | `python run_project.py --help` | Lists `--data`, `--eda`, `--train`, `--audit`, `--test`, `--edge`, `--dashboard`, and `--api`. |
| Streamlit dashboard | `python run_project.py --dashboard` | Local dashboard at `http://localhost:8501`. It has live/simulated audio monitor, 3D loom visualization, residual explainability, health trend, fleet matrix, and ROI views. |
| FastAPI service | `python run_project.py --api` | Local API at `http://localhost:8000`; OpenAPI UI at `/docs`; routes are `/health`, `/model`, and audio upload `/score`. |
| Dataset generation | `python scripts/generate_synthetic_dataset.py` | Rewrites `data/metadata.csv` and WAVs in `data/raw/synthetic/`. Do not run over the supplied dataset unless you first make a separate backup. |
| EDA | `python scripts/run_eda.py` | Writes figures and JSON sidecars `01` through `05` in `reports/figures/`; it can also generate the dataset if metadata is absent. Existing figures with the same names are overwritten. |
| Train/evaluate | `python scripts/train_and_evaluate.py` | Writes `reports/metrics/final_results.json`, report tables, and figures `06` through `09`; existing matching outputs are overwritten. Run only on a copy if preserving supplied results. |
| Edge timing benchmark | `python edge/benchmark.py` | Prints latency percentiles; does not write a report file. It benchmarks random noise on the current computer, not a Raspberry Pi. |
| Edge monitor demo | `python edge/record_and_score.py --machine-id LOOM-01 --max-cycles 5` | Prints simulated cycle scores and writes `edge_telemetry.csv` in the current directory, replacing that file if it exists. The script has no CLI option for a different output path. |
| Export utility | `python edge/export_onnx.py` | Rewrites `models/loomear_detector_manifest.json`. Despite its name, the current code writes a JSON manifest only; it does not create an ONNX file or save trained weights. |
| Notebooks | Open `notebooks/01_eda.ipynb` or `notebooks/02_cycles.ipynb` | Select the project environment kernel and run cells top-to-bottom. Both notebooks were unexecuted in the supplied files. Their path setup expects execution with the notebook working directory under `notebooks/`. |

`python run_project.py --all` is not a safe first command: it regenerates the dataset, rewrites EDA/training reports, runs the test/audit/edge scripts, and the edge demo overwrites `edge_telemetry.csv`. The Windows `Makefile` targets use `py` and `rm -rf`; use the Python commands above in PowerShell instead.

## Data, Configuration, and Interpretation Notes

- Verified from the supplied CSV: 144 rows, 48 each for `healthy`, `watch`, and `fault`; all rows say `synthetic`; all 144 metadata IDs have a corresponding WAV. There are 145 WAV files in the folder because the extra file is `ambience_loom_off.wav`.
- `reports/metrics/final_results.json`, README claims, and `docs/model_card.md` disagree with each other and with the current 144-row metadata (the stored manifest says 72 recordings, and several reported metrics differ). A fresh EDA/training run succeeded in a temporary project copy at `%TEMP%\LoomEar-verify-433245d4ac7e4dfebb84e244fc9db23b`; its metrics are not copied over the supplied reports.
- That fresh synthetic-only run reported PCA ROC AUC 1.000, One-Class SVM ROC AUC 1.000, LOMO ROC AUC 1.000, and XGBoost macro F1 0.9095. Do not interpret these synthetic benchmark results as real-mill performance. All reported partial AUCs were 0.5; the metric implementation returns that fallback when too few ROC points are available below the requested false-positive bound, so those values are not informative here.
- The training script writes its ablation scores as fixed constants rather than calculating ablation runs. Do not cite those entries as measured results.
- The main train/validation/test split separates sessions but not machines: all eight machine IDs occur in train and validation, and six occur in test. Machine exclusion is checked only in the separate LOMO split. Treat the main evaluation as session-held-out, not machine-held-out.
- Dashboard trend, fleet, and 3D views use generated or hard-coded demonstration data. They are not live machine telemetry. The edge monitor also generates audio in software rather than capturing a microphone.
- The dashboard's default `Normal Operational (Healthy)` simulation rendered a health score of 96.9 and `ACT` during browser verification. Its health baseline is not calibrated to the generated residual scale; treat the dashboard state as a demo defect, not a maintenance recommendation.
- The API computes a per-upload healthy template from the uploaded recording and chooses a fault label using a health-index threshold; it does not load a trained classifier. It now resamples uploads to 16 kHz (verified with an 8 kHz test). The API is unauthenticated and configured with permissive CORS; keep it local.
- No database or secret configuration was found. Dataset and reports use project-relative paths; start commands from the project root.

## Feature Verification Checklist

Status refers to execution in this environment, not code inspection alone.

| Feature/module | Source | Execute | Expected output | Actual result | Status |
|---|---|---|---|---|---|
| Metadata and audio inventory | `data/metadata.csv`, `data/raw/synthetic/` | CSV/WAV inventory check | Valid class/source counts and every referenced WAV present | 144 rows; 48 per condition; synthetic only; all 144 WAV references found | PASS |
| Python environment/dependencies | `requirements.txt` | Setup commands above | Usable isolated Python 3.11+ environment with imports installed | Requirements installed into a temporary Python 3.11.15 venv after retrying a network timeout; project-local venv creation failed | PASS |
| Unit/integration tests | `tests/` | `python -m pytest tests -v` | 17 test functions pass, including API upload | 17 passed; API score test covered both 16 kHz and 8 kHz; one Starlette/httpx deprecation warning | PASS |
| Leakage audit | `scripts/audit_leakage.py` | `python scripts/audit_leakage.py` | Session disjointness, LOMO machine isolation, valid source labels | All three audit checks passed on 144 synthetic rows. Additional check confirmed machines overlap in the main session split | PASS |
| Synthetic data generation | `scripts/generate_synthetic_dataset.py` | `python scripts/generate_synthetic_dataset.py` | Rebuilds WAVs and metadata | Not run; it would overwrite existing user data | BLOCKED |
| EDA and figures | `scripts/run_eda.py` | `python scripts/run_eda.py` | Five figures and sidecar JSON files | Completed on a temporary copy; figures `01`-`05` and JSON sidecars generated there | PASS |
| Model training and metrics | `scripts/train_and_evaluate.py` | `python scripts/train_and_evaluate.py` | Evaluation manifest, tables, and plots | Completed on a temporary copy; metrics/tables/figures generated there. Ablation values are hard-coded | PASS |
| Streamlit workflow | `app/dashboard.py` | `python run_project.py --dashboard` | Six interactive dashboard views | Launched at `http://localhost:8501`; all six views rendered in browser. Healthy simulation incorrectly displayed `ACT`, score 96.9 | FAIL |
| REST API routes and scoring | `src/loomear/serve/api.py` | `python run_project.py --api`; inspect `/docs` | Health/model JSON and uploaded audio score | `/health` and `/model` returned valid JSON; `/score` passed 16 kHz and 8 kHz integration cases | PASS |
| Edge benchmark | `edge/benchmark.py` | `python edge/benchmark.py` | Latency summary for synthetic noise | Ran on this Windows host: mean 9.13 ms, P50 9.01 ms, P95 10.68 ms per 3-second window; not a Pi benchmark | PASS |
| Edge monitor telemetry | `edge/record_and_score.py` | Command above | Simulated scores and telemetry CSV | Five simulated cycles completed; output written to `%TEMP%\loomear-edge-verify.csv`; original telemetry file untouched | PASS |
| ONNX export | `edge/export_onnx.py` | `python edge/export_onnx.py` | ONNX model and manifest | Ran in temporary copy: manifest was written, but no `.onnx` file was produced | FAIL |
| Jupyter notebooks | `notebooks/01_eda.ipynb`, `notebooks/02_cycles.ipynb` | Run cells in VS Code | EDA and cycle/residual plots | Both notebooks are present but were not executed | BLOCKED |

## Outputs and Changes

Existing outputs are under `reports/figures/`, `reports/metrics/`, `reports/tables/`, and `edge_telemetry.csv`. No dataset, report, or telemetry output was intentionally regenerated in this inspection.

Files changed for this setup work:

- `requirements.txt`: declared Librosa, Plotly, `python-multipart`, and Tabulate, which existing code imports or requires.
- `src/loomear/health/cost_model.py`: imported the missing `Optional` type that prevented test collection.
- `src/loomear/eval/explain.py`: made SHAP optional at import time so API startup does not require the separate explainer package.
- `src/loomear/serve/api.py` and `tests/test_api.py`: resampled non-16 kHz API uploads and added 8 kHz coverage.
- `README_RUN.md`: this project-specific Windows setup, execution, safety, and verification guide.
- `.vscode/tasks.json`: added a portable test task using the documented `%TEMP%` environment path.

The inspected environment also has an incomplete ignored `.venv` directory from the failed venv creation attempt. It was left in place rather than deleted.