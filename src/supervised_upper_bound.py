from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataframe
from evaluate import calibrate_threshold, compute_metrics, format_metrics
from experiment_utils import (
    append_csv,
    append_jsonl,
    load_config,
    set_global_seed,
    utc_timestamp,
    write_json,
)
from preprocess import (
    build_data_quality_summary,
    build_features_and_labels_with_report,
    clean_dataframe,
    split_train_val_test_for_anomaly,
    validate_schema,
)

MODEL_NAME = "supervised_rf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised upper bound: RandomForest trained on fully labeled data."
    )
    parser.add_argument("--config", type=str, default="configs/phase3_contrastive.json")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
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
    val_size = float(cfg.get("val_size", 0.1))
    test_size = float(cfg.get("test_size", 0.3))
    threshold_quantile = float(cfg.get("threshold_quantile", 0.95))
    sample_rows = args.sample_rows if args.sample_rows is not None else int(cfg.get("sample_rows", 30000))
    random_seed = args.random_seed if args.random_seed is not None else int(cfg.get("random_seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    output_dir = cfg.get("output_dir", "results")
    run_log_jsonl = cfg.get("run_log_jsonl", "results/experiment_runs.jsonl")
    run_log_csv = cfg.get("run_log_csv", "results/experiment_runs.csv")

    n_estimators = 200
    max_features = "sqrt"

    set_global_seed(seed=random_seed, deterministic=deterministic)

    print("[1/5] Loading data...")
    df = load_dataframe(data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)
        print(f"  Downsampled to {len(df)} rows.")

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

    print("[3/5] Splitting data (same protocol as other models)...")
    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(
        X, y, val_size=val_size, test_size=test_size, random_state=random_seed
    )

    # Supervised training pool = X_train (all normal) + X_val (normal + anomaly).
    # X_train contains only normal traffic, so its labels are all zeros.
    y_train_normal = pd.Series(
        np.zeros(len(X_train), dtype=int), index=X_train.index
    )
    X_sup = pd.concat([X_train, X_val], axis=0)
    y_sup = pd.concat([y_train_normal, y_val], axis=0)

    scaler = StandardScaler()
    X_sup_scaled = scaler.fit_transform(X_sup)
    X_test_scaled = scaler.transform(X_test)

    n_attack_in_train = int(y_sup.sum())
    print(
        f"  Supervised training set: {len(X_sup)} samples "
        f"({n_attack_in_train} attack, {len(X_sup) - n_attack_in_train} normal)"
    )

    print("[4/5] Training RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        oob_score=True,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_sup_scaled, y_sup.to_numpy())

    # Threshold is calibrated on out-of-bag predictions so no training sample
    # is scored by a tree that saw it — equivalent to a held-out validation set.
    oob_scores = model.oob_decision_function_[:, 1]
    threshold = calibrate_threshold(oob_scores, threshold_quantile)

    test_scores = model.predict_proba(X_test_scaled)[:, 1]

    np.savez(
        Path(output_dir) / "supervised_rf_scores.npz",
        scores=test_scores,
        y_true=y_test.to_numpy(),
    )

    print("[5/5] Evaluating...")
    metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        anomaly_scores=test_scores,
        threshold_quantile=threshold_quantile,
        threshold=threshold,
    )
    print(format_metrics(metrics))

    run_id = f"{MODEL_NAME}_{utc_timestamp()}"
    payload = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "random_seed": random_seed,
        "sample_rows": sample_rows,
        "val_size": val_size,
        "test_size": test_size,
        "threshold_quantile": threshold_quantile,
        "threshold_source": "oob_predictions",
        "n_estimators": n_estimators,
        "max_features": max_features,
        "train_rows": int(len(X_sup)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "val_anomalies": int(y_val.sum()),
        "test_anomalies": int(y_test.sum()),
        "supervised_train_anomalies": n_attack_in_train,
        "schema_report": schema_report,
        "preprocess_report": preprocess_report,
        "quality_summary": quality_summary,
        **metrics,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    latest_path = output_path / "supervised_rf_metrics.json"
    run_path = output_path / "metrics" / f"{run_id}.json"

    write_json(latest_path, payload)
    write_json(run_path, payload)
    append_jsonl(run_log_jsonl, payload)
    append_csv(run_log_csv, payload)

    print(f"Metrics saved → {latest_path}")


if __name__ == "__main__":
    main()
