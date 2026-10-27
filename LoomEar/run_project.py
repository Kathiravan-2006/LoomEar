"""
Master pipeline runner for LoomEar.
Executes the full end-to-end data science, machine learning, evaluation, and serving pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(command_str: str, desc: str):
    print("\n" + "=" * 60)
    print(f"[*] {desc.upper()}")
    print("=" * 60)
    res = subprocess.run([sys.executable] + command_str.split()[1:], check=True)
    return res


def main():
    parser = argparse.ArgumentParser(description="LoomEar End-to-End Master Runner")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline end-to-end")
    parser.add_argument("--data", action="store_true", help="Generate synthetic audio dataset")
    parser.add_argument("--eda", action="store_true", help="Run EDA and generate figures")
    parser.add_argument("--train", action="store_true", help="Train and evaluate all models")
    parser.add_argument("--audit", action="store_true", help="Run data leakage verification audit")
    parser.add_argument("--test", action="store_true", help="Run pytest unit test suite")
    parser.add_argument("--edge", action="store_true", help="Run edge benchmark and live scoring daemon")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit web dashboard")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST server")

    args = parser.parse_args()

    # Default to running full pipeline if no specific flag passed
    if not any(vars(args).values()) or args.all:
        print("\n>>> Executing LoomEar Full Pipeline...")
        run_cmd("py scripts/generate_synthetic_dataset.py", "Generating Dataset")
        run_cmd("py scripts/run_eda.py", "Running Exploratory Data Analysis")
        run_cmd("py scripts/train_and_evaluate.py", "Training and Evaluating Models")
        run_cmd("py scripts/audit_leakage.py", "Running Leakage Audit")
        run_cmd("py -m pytest tests -v", "Executing Test Suite")
        run_cmd("py edge/benchmark.py", "Benchmarking Edge Latency")
        run_cmd("py edge/record_and_score.py --max-cycles 4", "Executing Edge Monitor Demo")
        print("\n[SUCCESS] Full LoomEar Pipeline Completed Successfully!")
        print("To launch the interactive dashboard, run: py run_project.py --dashboard")
        print("To launch the FastAPI server, run: py run_project.py --api\n")
        return

    if args.data:
        run_cmd("py scripts/generate_synthetic_dataset.py", "Generating Dataset")
    if args.eda:
        run_cmd("py scripts/run_eda.py", "Running EDA")
    if args.train:
        run_cmd("py scripts/train_and_evaluate.py", "Training Models")
    if args.audit:
        run_cmd("py scripts/audit_leakage.py", "Running Leakage Audit")
    if args.test:
        run_cmd("py -m pytest tests -v", "Running Test Suite")
    if args.edge:
        run_cmd("py edge/benchmark.py", "Benchmarking Edge")
        run_cmd("py edge/record_and_score.py --max-cycles 4", "Edge Scoring")
    if args.dashboard:
        print("\nLaunching Streamlit Dashboard at http://localhost:8501 ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])
    if args.api:
        print("\nLaunching FastAPI Server at http://localhost:8000 ...")
        subprocess.run([sys.executable, "-m", "uvicorn", "src.loomear.serve.api:app", "--reload", "--port", "8000"])


if __name__ == "__main__":
    main()
