from __future__ import annotations

import argparse
from pathlib import Path

from nta.data_loader import load_dataframe
from nta.experiment_utils import load_config
from nta.preprocess import (
    build_data_quality_summary,
    build_features_and_labels_with_report,
    clean_dataframe,
    split_train_val_test_for_anomaly,
    validate_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data quality and split protocol report.")
    parser.add_argument("--config", type=str, default="configs/baselines.json", help="Path to config JSON")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/data_pipeline_report.md",
        help="Path to output markdown report",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Optional dataset path override")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sample size override")
    return parser.parse_args()


def bullet_list(items: list[str], default_text: str = "None") -> str:
    if not items:
        return default_text
    return ", ".join(items)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data_path if args.data_path is not None else cfg.get("data_path")
    label_column = cfg.get("label_column", "Label")
    required_columns = cfg.get("required_columns", [])
    min_rows = int(cfg.get("min_rows", 100))
    max_missing_ratio = float(cfg.get("max_missing_ratio", 0.5))
    leakage_patterns = cfg.get("leakage_patterns", None)
    val_size = float(cfg.get("val_size", 0.1))
    test_size = float(cfg.get("test_size", 0.3))
    sample_rows = args.sample_rows if args.sample_rows is not None else int(cfg.get("sample_rows", 50000))
    random_seed = int(cfg.get("random_seed", 42))

    df = load_dataframe(data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)

    df = clean_dataframe(df)

    schema_report = validate_schema(
        df,
        label_column=label_column,
        required_columns=required_columns,
        min_rows=min_rows,
    )
    X, y, preprocess_report = build_features_and_labels_with_report(
        df,
        label_column=label_column,
        max_missing_ratio=max_missing_ratio,
        leakage_patterns=leakage_patterns,
    )
    quality_summary = build_data_quality_summary(df)

    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(
        X,
        y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_seed,
    )

    markdown = []
    markdown.append("# Data Pipeline Report")
    markdown.append("")
    markdown.append("## Schema Validation")
    markdown.append("")
    markdown.append(f"- Rows: {schema_report['n_rows']}")
    markdown.append(f"- Columns: {schema_report['n_columns']}")
    markdown.append(f"- Duplicate rows: {schema_report['duplicate_rows']}")
    markdown.append(f"- Global missing ratio: {schema_report['global_missing_ratio']:.4f}")
    markdown.append(f"- Label distribution: {schema_report['label_distribution']}")
    markdown.append("")
    markdown.append("## Feature Engineering and Leakage Checks")
    markdown.append("")
    markdown.append(f"- Original feature columns: {preprocess_report['original_feature_columns']}")
    markdown.append(f"- Final feature columns: {preprocess_report['final_feature_columns']}")
    markdown.append(
        "- Dropped leakage columns: "
        + bullet_list(preprocess_report["dropped_leakage_columns"], default_text="None detected")
    )
    markdown.append(
        "- Dropped high-missing columns: "
        + bullet_list(preprocess_report["dropped_high_missing_columns"], default_text="None")
    )
    markdown.append(
        "- Dropped constant columns: "
        + bullet_list(preprocess_report["dropped_constant_columns"], default_text="None")
    )
    markdown.append("")
    markdown.append("## Train/Validation/Test Protocol")
    markdown.append("")
    markdown.append(f"- Train rows (normal only): {len(X_train)}")
    markdown.append(f"- Validation rows: {len(X_val)} (anomalies: {int(y_val.sum())})")
    markdown.append(f"- Test rows: {len(X_test)} (anomalies: {int(y_test.sum())})")
    markdown.append(f"- Validation size: {val_size}")
    markdown.append(f"- Test size: {test_size}")
    markdown.append("")
    markdown.append("## Data Quality Snapshot")
    markdown.append("")
    markdown.append(f"- Constant columns count: {quality_summary['constant_columns_count']}")
    markdown.append(f"- Top missing columns: {quality_summary['top_missing_columns']}")
    markdown.append("")
    markdown.append("Data pipeline checks and split protocol are active and reproducible via config.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(f"Saved data pipeline report to: {output_path}")


if __name__ == "__main__":
    main()
