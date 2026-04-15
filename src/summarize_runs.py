from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline runs into a comparison table.")
    parser.add_argument("--runs-jsonl", type=str, default="results/experiment_runs.jsonl", help="Path to experiment run JSONL")
    parser.add_argument("--output", type=str, default="reports/phase1_baseline_report.md", help="Output markdown report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_path = Path(args.runs_jsonl)
    if not runs_path.exists():
        raise FileNotFoundError(f"Run log JSONL not found: {runs_path}")

    records = []
    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Run log JSONL is empty.")

    latest_per_model = (
        df.sort_values("run_id")
        .groupby("model", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    cols = ["model", "roc_auc", "pr_auc", "precision", "recall", "f1", "fpr", "n_samples", "n_anomalies"]
    missing = [c for c in cols if c not in latest_per_model.columns]
    if missing:
        raise ValueError(f"Missing required columns in run log: {missing}")

    table = latest_per_model[cols].copy()
    for metric_col in ["roc_auc", "pr_auc", "precision", "recall", "f1", "fpr"]:
        table[metric_col] = table[metric_col].map(lambda x: f"{x:.4f}")

    best_model = latest_per_model.sort_values("f1", ascending=False).iloc[0]

    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    markdown_table = "\n".join([header, divider, *rows])

    markdown = []
    markdown.append("# Phase 1 Experiment Report")
    markdown.append("")
    markdown.append("## Baseline Comparison Table")
    markdown.append("")
    markdown.append(markdown_table)
    markdown.append("")
    markdown.append("## First Clean Experiment Report Section")
    markdown.append("")
    markdown.append("We compared two unsupervised baselines trained on normal traffic and evaluated on held-out normal plus anomalous traffic.")
    markdown.append("The current best F1 among latest runs is from model: " + str(best_model["model"]) + ".")
    markdown.append(
        "This phase establishes a reproducible benchmark so self-supervised methods can be evaluated against a stable baseline in the next phase."
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown), encoding="utf-8")

    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
