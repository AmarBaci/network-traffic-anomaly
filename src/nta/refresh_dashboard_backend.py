from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_STEPS = [
    "nta.data_quality",
    "nta.baseline_isolation_forest",
    "nta.baseline_one_class_svm",
    "nta.contrastive_ssl",
    "nta.summarize_runs",
    "nta.export_dashboard_data",
]

SKIP_TRAIN_STEPS = [
    "nta.data_quality",
    "nta.summarize_runs",
    "nta.export_dashboard_data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full backend refresh for dashboard consumption.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training steps and refresh dashboard from existing logs")
    return parser.parse_args()


def run_step(module: str) -> None:
    cmd = [sys.executable, "-m", module]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    steps = SKIP_TRAIN_STEPS if args.skip_train else DEFAULT_STEPS
    for step in steps:
        run_step(step)

    print("Dashboard backend refresh completed successfully.")


if __name__ == "__main__":
    main()
