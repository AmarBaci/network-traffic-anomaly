from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_loader import load_dataframe
from evaluate import compute_metrics, format_metrics
from preprocess import build_features_and_labels, clean_dataframe, split_train_test_for_anomaly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 baseline: Isolation Forest on CIC-style traffic data."
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV or ZIP dataset")
    parser.add_argument("--label-column", type=str, default="Label", help="Label column name")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction of normal traffic held out for test")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly ratio for IsolationForest")
    parser.add_argument("--threshold-quantile", type=float, default=0.95, help="Quantile threshold for anomaly decision")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Optional downsampling for fast first run")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to write metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/5] Loading data...")
    df = load_dataframe(args.data_path)

    if args.sample_rows and args.sample_rows < len(df):
        df = df.sample(n=args.sample_rows, random_state=args.random_seed).reset_index(drop=True)
        print(f"Downsampled to {len(df)} rows for quick iteration.")

    print("[2/5] Cleaning and building features...")
    df = clean_dataframe(df)
    X, y = build_features_and_labels(df, label_column=args.label_column)

    print("[3/5] Splitting train/test for anomaly setting...")
    X_train, X_test, y_test = split_train_test_for_anomaly(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[4/5] Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=args.contamination,
        random_state=args.random_seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    # Higher values should indicate more anomalous points.
    anomaly_scores = -model.score_samples(X_test_scaled)

    print("[5/5] Evaluating...")
    metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        anomaly_scores=anomaly_scores,
        threshold_quantile=args.threshold_quantile,
    )
    print(format_metrics(metrics))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_isolation_forest_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
