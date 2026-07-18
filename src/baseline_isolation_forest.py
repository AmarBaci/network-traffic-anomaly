from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataframe
from evaluate import calibrate_threshold, compute_metrics, format_metrics
from experiment_utils import append_csv, append_jsonl, load_config, set_global_seed, utc_timestamp, write_json
from preprocess import (
    build_data_quality_summary,
    build_features_and_labels_with_report,
    clean_dataframe,
    split_train_val_test_for_anomaly,
    validate_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline: Isolation Forest on CIC-style traffic data."
    )
    parser.add_argument("--config", type=str, default="configs/baselines.json", help="Path to config JSON")
    parser.add_argument("--data-path", type=str, default=None, help="Optional dataset path override")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sample size override")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional seed override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data_path if args.data_path is not None else cfg.get("data_path")
    label_column = cfg.get("label_column", "Label")
    required_columns = cfg.get("required_columns", [])
    min_rows = int(cfg.get("min_rows", 100))
    max_missing_ratio = float(cfg.get("max_missing_ratio", 0.5))
    leakage_patterns = cfg.get("leakage_patterns", None)
    test_size = float(cfg.get("test_size", 0.3))
    threshold_quantile = float(cfg.get("threshold_quantile", 0.95))
    val_size = float(cfg.get("val_size", 0.1))
    sample_rows = args.sample_rows if args.sample_rows is not None else int(cfg.get("sample_rows", 50000))
    random_seed = args.random_seed if args.random_seed is not None else int(cfg.get("random_seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    output_dir = cfg.get("output_dir", "results")
    run_log_jsonl = cfg.get("run_log_jsonl", "results/experiment_runs.jsonl")
    run_log_csv = cfg.get("run_log_csv", "results/experiment_runs.csv")

    model_cfg = cfg.get("isolation_forest", {})
    contamination = float(model_cfg.get("contamination", 0.05))
    n_estimators = int(model_cfg.get("n_estimators", 200))

    set_global_seed(seed=random_seed, deterministic=deterministic)

    print("[1/5] Loading data...")
    df = load_dataframe(data_path)

    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)
        print(f"Downsampled to {len(df)} rows for quick iteration.")

    print("[2/5] Cleaning and building features...")
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

    print("[3/5] Splitting train/test for anomaly setting...")
    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(
        X,
        y,
        val_size=val_size,
        test_size=test_size,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("[4/5] Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    val_scores = -model.score_samples(X_val_scaled)
    threshold = calibrate_threshold(val_scores, threshold_quantile)

    # Higher values should indicate more anomalous points.
    anomaly_scores = -model.score_samples(X_test_scaled)

    np.savez(
        Path(output_dir) / "baseline_isolation_forest_scores.npz",
        scores=anomaly_scores,
        y_true=y_test.to_numpy(),
    )

    print("[5/5] Evaluating...")
    metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        anomaly_scores=anomaly_scores,
        threshold_quantile=threshold_quantile,
        threshold=threshold,
    )
    print(format_metrics(metrics))

    run_id = f"isolation_forest_{utc_timestamp()}"
    metrics_payload = {
        "run_id": run_id,
        "model": "isolation_forest",
        "random_seed": random_seed,
        "sample_rows": sample_rows,
        "val_size": val_size,
        "test_size": test_size,
        "threshold_quantile": threshold_quantile,
        "threshold_source": "validation_scores",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "val_anomalies": int(y_val.sum()),
        "test_anomalies": int(y_test.sum()),
        "schema_report": schema_report,
        "preprocess_report": preprocess_report,
        "quality_summary": quality_summary,
        **metrics,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    latest_metrics_path = output_path / "baseline_isolation_forest_metrics.json"
    run_metrics_path = output_path / "metrics" / f"{run_id}.json"

    write_json(latest_metrics_path, metrics_payload)
    write_json(run_metrics_path, metrics_payload)
    append_jsonl(run_log_jsonl, metrics_payload)
    append_csv(run_log_csv, metrics_payload)

    print(f"Saved latest metrics to: {latest_metrics_path}")
    print(f"Saved run metrics to: {run_metrics_path}")
    print(f"Appended run logs to: {run_log_jsonl} and {run_log_csv}")


if __name__ == "__main__":
    main()
