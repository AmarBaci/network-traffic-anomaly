from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STEPS = [
    "src/phase2_data_quality.py",
    "src/baseline_isolation_forest.py",
    "src/baseline_one_class_svm.py",
    "src/phase3_contrastive_model.py",
    "src/summarize_runs.py",
    "src/export_dashboard_data.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full backend refresh for dashboard consumption.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training steps and refresh dashboard from existing logs")
    return parser.parse_args()


def run_step(script_path: str) -> None:
    cmd = [sys.executable, script_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    steps = list(DEFAULT_STEPS)
    if args.skip_train:
        steps = [
            "src/phase2_data_quality.py",
            "src/summarize_runs.py",
            "src/export_dashboard_data.py",
        ]

    for step in steps:
        if not Path(step).exists():
            raise FileNotFoundError(f"Missing script step: {step}")
        run_step(step)

    print("Dashboard backend refresh completed successfully.")


if __name__ == "__main__":
    main()
