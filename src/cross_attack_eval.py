"""Evaluate the trained SSL model on a new CIC-IDS attack file.

The model (weights + normal-traffic centroid) was trained on DDoS traffic.
This script tests whether the learned normal-traffic representation generalises
to attack types not seen during training (port scan, botnet, web attacks, etc.).

Usage:
    python src/cross_attack_eval.py --new-data data/<attack_file>.csv.zip

Download additional CIC-IDS2017 files from:
    https://www.kaggle.com/datasets/cicdataset/cicids2017
    (Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,
     Friday-WorkingHours-Morning.pcap_ISCX.csv  [Botnet],
     Wednesday-workingHours.pcap_ISCX.csv       [DoS],
     Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from contrastive_net import ContrastiveAutoencoder
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
    clean_dataframe,
    split_train_val_test_for_anomaly,
    validate_schema,
)

METADATA_COLUMNS = {
    "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "SimillarHTTP", "Inbound",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-attack evaluation: apply trained SSL model to a new CIC-IDS file."
    )
    parser.add_argument(
        "--new-data", required=True,
        help="Path to the new CIC-IDS CSV or ZIP file to evaluate on.",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/phase3_ssl_checkpoint.pt",
        help="Path to the trained SSL checkpoint (default: results/phase3_ssl_checkpoint.pt).",
    )
    parser.add_argument(
        "--metrics-json",
        default="results/phase3_contrastive_ssl_metrics.json",
        help="Path to the training metrics JSON (used to recover dropped columns).",
    )
    parser.add_argument(
        "--config",
        default="configs/phase3_contrastive.json",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=30000,
        help="Rows to sample from the new file (default: 30000).",
    )
    parser.add_argument(
        "--label-column", type=str, default="Label",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
    )
    return parser.parse_args()


def _score(
    model: ContrastiveAutoencoder,
    X_tensor: torch.Tensor,
    centroid: torch.Tensor,
    device: torch.device,
    recon_weight: float = 0.5,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            h, _, x_hat = model(batch)
            dist = torch.norm(h - centroid, dim=-1)
            recon = F.mse_loss(x_hat, batch, reduction="none").mean(dim=-1)
            parts.append(
                ((1.0 - recon_weight) * dist + recon_weight * recon).cpu().numpy()
            )
    return np.concatenate(parts)


def reconstruct_scaler(
    original_data_path: str,
    label_column: str,
    constant_cols: list[str],
    sample_rows: int,
    random_seed: int,
    val_size: float,
    test_size: float,
) -> tuple[StandardScaler, list[str]]:
    """Re-run original preprocessing with the same seed to recover the fitted scaler.

    Because data loading and splitting are deterministic, this produces an
    identical scaler to the one used during training without re-training the model.
    """
    print("  Loading original training data to reconstruct scaler...")
    df = load_dataframe(original_data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)

    df = clean_dataframe(df)

    labels = df[label_column].astype(str).str.strip().str.lower()
    y = (labels != "benign").astype(int)

    cols_to_drop = {label_column} | METADATA_COLUMNS | set(constant_cols)
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])

    X_train, _, _, _, _ = split_train_val_test_for_anomaly(
        X, y, val_size=val_size, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler, list(X_train.columns)


def load_new_file(
    data_path: str,
    label_column: str,
    constant_cols: list[str],
    feature_columns: list[str],
    scaler: StandardScaler,
    sample_rows: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and preprocess a new attack file, aligning it to the original feature set."""
    df = load_dataframe(data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)

    df = clean_dataframe(df)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns[:10])}")

    labels = df[label_column].astype(str).str.strip().str.lower()
    y = (labels != "benign").astype(int)

    label_dist = labels.value_counts().to_dict()
    n_normal   = int((y == 0).sum())
    n_anomaly  = int((y == 1).sum())

    cols_to_drop = {label_column} | METADATA_COLUMNS | set(constant_cols)
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])

    # Align to the exact feature set the model was trained on.
    missing = [c for c in feature_columns if c not in X.columns]
    extra   = [c for c in X.columns if c not in feature_columns]
    if missing:
        print(f"  Warning: {len(missing)} features missing from new file — filling with 0: {missing[:5]}")
        for col in missing:
            X[col] = 0.0
    if extra:
        X = X.drop(columns=extra)
    X = X[feature_columns]  # enforce original column order

    X_scaled = scaler.transform(X).astype(np.float32)

    info = {
        "n_rows": len(df),
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "label_distribution": label_dist,
        "missing_features": missing,
    }
    return X_scaled, y.to_numpy(), info


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    new_data_path = args.new_data
    if not Path(new_data_path).exists():
        raise FileNotFoundError(
            f"New data file not found: {new_data_path}\n"
            "Download CIC-IDS2017 files from https://www.kaggle.com/datasets/cicdataset/cicids2017"
        )

    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run phase3_contrastive_model.py first.")

    metrics_path = args.metrics_json
    if not Path(metrics_path).exists():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}.")

    with open(metrics_path) as f:
        training_metrics = json.load(f)

    preprocess_report  = training_metrics["preprocess_report"]
    constant_cols      = preprocess_report["dropped_constant_columns"]
    original_data_path = cfg.get("data_path") or "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.zip"
    val_size           = float(training_metrics.get("val_size", 0.1))
    test_size          = float(training_metrics.get("test_size", 0.3))
    original_seed      = int(training_metrics.get("random_seed", 42))
    original_rows      = int(training_metrics.get("sample_rows", 30000))
    threshold_quantile = float(training_metrics.get("threshold_quantile", 0.95))

    set_global_seed(seed=original_seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ model
    print("[1/4] Loading trained model and centroid from checkpoint...")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ContrastiveAutoencoder(
        input_dim  = ck["input_dim"],
        hidden_dim = ck["hidden_dim"],
        embed_dim  = ck["embed_dim"],
        proj_dim   = ck["proj_dim"],
    ).to(device)
    model.load_state_dict(ck["model_state"])
    centroid = ck["centroid"].to(device)
    recon_weight = float(cfg.get("contrastive", {}).get("recon_weight", 0.5))
    print(f"  input_dim={ck['input_dim']}  embed_dim={ck['embed_dim']}")

    # ------------------------------------------------------------------ scaler
    print("[2/4] Reconstructing original StandardScaler...")
    scaler, feature_columns = reconstruct_scaler(
        original_data_path=original_data_path,
        label_column=args.label_column,
        constant_cols=constant_cols,
        sample_rows=original_rows,
        random_seed=original_seed,
        val_size=val_size,
        test_size=test_size,
    )
    print(f"  Scaler reconstructed on {len(feature_columns)} features.")

    # ------------------------------------------------------------------ new file
    print(f"[3/4] Loading new attack file: {new_data_path}")
    X_new, y_new, file_info = load_new_file(
        data_path      = new_data_path,
        label_column   = args.label_column,
        constant_cols  = constant_cols,
        feature_columns= feature_columns,
        scaler         = scaler,
        sample_rows    = args.sample_rows,
        random_seed    = original_seed,
    )
    print(f"  Rows: {file_info['n_rows']}  |  Normal: {file_info['n_normal']}  |  Attack: {file_info['n_anomaly']}")
    print(f"  Attack types: {list(file_info['label_distribution'].keys())}")

    if file_info["n_anomaly"] == 0:
        print("  Warning: no anomaly rows found in this file — metrics will not be meaningful.")

    # ------------------------------------------------------------------ score
    print("[4/4] Scoring and evaluating...")
    X_tensor = torch.from_numpy(X_new)

    # Use normal rows only to calibrate the threshold (mirrors training protocol).
    normal_mask = y_new == 0
    if normal_mask.sum() > 0:
        X_normal_t = X_tensor[normal_mask]
        normal_scores = _score(model, X_normal_t, centroid, device, recon_weight)
        threshold = calibrate_threshold(normal_scores, threshold_quantile)
    else:
        # Fallback: calibrate on full file scores if no normal rows.
        all_scores = _score(model, X_tensor, centroid, device, recon_weight)
        threshold = calibrate_threshold(all_scores, threshold_quantile)

    test_scores = _score(model, X_tensor, centroid, device, recon_weight)
    metrics = compute_metrics(
        y_true=y_new,
        anomaly_scores=test_scores,
        threshold_quantile=threshold_quantile,
        threshold=threshold,
    )
    print(format_metrics(metrics))

    # ------------------------------------------------------------------ save
    attack_name = Path(new_data_path).stem.replace(".pcap_ISCX", "").replace(".csv", "")
    run_id      = f"cross_attack_{attack_name}_{utc_timestamp()}"
    payload = {
        "run_id":             run_id,
        "model":              "cross_attack_ssl",
        "attack_file":        str(new_data_path),
        "attack_name":        attack_name,
        "label_distribution": {str(k): int(v) for k, v in file_info["label_distribution"].items()},
        "n_normal":           file_info["n_normal"],
        "n_anomaly":          file_info["n_anomaly"],
        "threshold_quantile": threshold_quantile,
        "threshold_source":   "normal_rows_in_new_file",
        "checkpoint":         str(checkpoint_path),
        **metrics,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"cross_attack_{attack_name}_metrics.json"
    write_json(result_path, payload)

    run_log_jsonl = cfg.get("run_log_jsonl", "results/experiment_runs.jsonl")
    run_log_csv   = cfg.get("run_log_csv",   "results/experiment_runs.csv")
    append_jsonl(run_log_jsonl, payload)
    append_csv(run_log_csv,    payload)

    print(f"\nResults saved → {result_path}")


if __name__ == "__main__":
    main()
