from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from nta.contrastive_augment import TabularAugmentation
from nta.contrastive_net import ContrastiveAutoencoder, nt_xent_loss
from nta.data_loader import load_dataframe
from nta.evaluate import calibrate_threshold, compute_metrics
from nta.experiment_utils import (
    append_csv,
    append_jsonl,
    load_config,
    set_global_seed,
    utc_timestamp,
    write_json,
)
from nta.preprocess import (
    build_features_and_labels_with_report,
    clean_dataframe,
    split_train_val_test_for_anomaly,
    validate_schema,
)

# Each entry varies exactly one axis vs the full model so the contribution
# of each component can be read off directly from the results table.
ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "full_model",
        "label": "Full model (NT-Xent + Recon, combined aug)",
        "contrastive_weight": 0.5,
        "recon_weight": 0.5,
        "strategy": "combined",
    },
    {
        "name": "no_reconstruction",
        "label": "No reconstruction loss (NT-Xent only)",
        "contrastive_weight": 1.0,
        "recon_weight": 0.0,
        "strategy": "combined",
    },
    {
        "name": "no_contrastive",
        "label": "No contrastive loss (reconstruction only)",
        "contrastive_weight": 0.0,
        "recon_weight": 1.0,
        "strategy": "combined",
    },
    {
        "name": "noise_only",
        "label": "Noise augmentation only",
        "contrastive_weight": 0.5,
        "recon_weight": 0.5,
        "strategy": "noise",
    },
    {
        "name": "mask_only",
        "label": "Mask augmentation only",
        "contrastive_weight": 0.5,
        "recon_weight": 0.5,
        "strategy": "mask",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study: isolate the contribution of each SSL component."
    )
    parser.add_argument("--config", type=str, default="configs/contrastive.json")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per ablation run (default 50; use 100 to match full training).",
    )
    return parser.parse_args()


def _compute_centroid(
    model: ContrastiveAutoencoder,
    X_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    model.eval()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            parts.append(model.encode(batch))
    return torch.cat(parts).mean(dim=0)


def _score(
    model: ContrastiveAutoencoder,
    X_tensor: torch.Tensor,
    centroid: torch.Tensor,
    device: torch.device,
    recon_weight: float,
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


def run_ablation(
    name: str,
    contrastive_weight: float,
    recon_weight: float,
    strategy: str,
    X_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_val: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    device: torch.device,
    cfg: dict[str, Any],
    epochs: int,
    random_seed: int,
) -> dict[str, Any]:
    mcfg = cfg.get("contrastive", {})
    acfg = cfg.get("augmentation", {})
    threshold_quantile = float(cfg.get("threshold_quantile", 0.95))

    hidden_dim = int(mcfg.get("hidden_dim", 128))
    embed_dim = int(mcfg.get("embed_dim", 64))
    proj_dim = int(mcfg.get("proj_dim", 32))
    dropout = float(mcfg.get("dropout", 0.2))
    learning_rate = float(mcfg.get("learning_rate", 1e-3))
    weight_decay = float(mcfg.get("weight_decay", 1e-4))
    batch_size = int(mcfg.get("batch_size", 256))
    temperature = float(mcfg.get("temperature", 0.5))

    noise_std = float(acfg.get("noise_std", 0.05))
    mask_prob = float(acfg.get("mask_prob", 0.15))
    jitter_range = tuple(acfg.get("jitter_range", [0.8, 1.2]))

    set_global_seed(seed=random_seed, deterministic=True)

    model = ContrastiveAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        dropout=dropout,
    ).to(device)

    augment = TabularAugmentation(
        noise_std=noise_std,
        mask_prob=mask_prob,
        jitter_range=jitter_range,
        strategy=strategy,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(X_train_t), batch_size=batch_size, shuffle=True, drop_last=True
    )

    model.train()
    for epoch in range(1, epochs + 1):
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x1, x2 = augment(x_batch)
            h1, z1, x1_hat = model(x1)
            h2, z2, x2_hat = model(x2)
            loss_c = nt_xent_loss(z1, z2, temperature=temperature)
            loss_r = 0.5 * (F.mse_loss(x1_hat, x_batch) + F.mse_loss(x2_hat, x_batch))
            loss = contrastive_weight * loss_c + recon_weight * loss_r
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    centroid = _compute_centroid(model, X_train_t, device)
    val_scores = _score(model, X_val_t, centroid, device, recon_weight)
    threshold = calibrate_threshold(val_scores, threshold_quantile)
    test_scores = _score(model, X_test_t, centroid, device, recon_weight)

    return compute_metrics(
        y_true=y_test,
        anomaly_scores=test_scores,
        threshold_quantile=threshold_quantile,
        threshold=threshold,
    )


def _fmt_row(label: str, m: dict[str, Any]) -> str:
    return (
        f"  {label:<48}"
        f"  {m['roc_auc']:.4f}"
        f"  {m['pr_auc']:.4f}"
        f"  {m['f1']:.4f}"
        f"  {m['fpr']:.4f}"
    )


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
    sample_rows = args.sample_rows if args.sample_rows is not None else int(cfg.get("sample_rows", 30000))
    random_seed = args.random_seed if args.random_seed is not None else int(cfg.get("random_seed", 42))
    output_dir = cfg.get("output_dir", "results")
    run_log_jsonl = cfg.get("run_log_jsonl", "results/experiment_runs.jsonl")
    run_log_csv = cfg.get("run_log_csv", "results/experiment_runs.csv")

    set_global_seed(seed=random_seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}  |  Epochs per run: {args.epochs}\n")

    # ------------------------------------------------------------------ data
    print("[data] Loading and preprocessing (shared across all runs)...")
    df = load_dataframe(data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)

    df = clean_dataframe(df)
    validate_schema(df, label_column=label_column, required_columns=required_columns, min_rows=min_rows)
    X, y, _ = build_features_and_labels_with_report(
        df, label_column=label_column, max_missing_ratio=max_missing_ratio,
        leakage_patterns=leakage_patterns,
    )

    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(
        X, y, val_size=val_size, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_t = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32))
    X_val_t   = torch.from_numpy(scaler.transform(X_val).astype(np.float32))
    X_test_t  = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
    input_dim = X_train_t.shape[1]
    y_val_np  = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    print(f"  {len(X_train)} train (normal only) | {len(X_val)} val | {len(X_test)} test\n")

    # ------------------------------------------------------------------ runs
    results: list[dict[str, Any]] = []

    for i, ablation in enumerate(ABLATION_CONFIGS, start=1):
        name     = ablation["name"]
        label    = ablation["label"]
        c_weight = ablation["contrastive_weight"]
        r_weight = ablation["recon_weight"]
        strategy = ablation["strategy"]

        print(f"[{i}/{len(ABLATION_CONFIGS)}] {label}")

        metrics = run_ablation(
            name=name,
            contrastive_weight=c_weight,
            recon_weight=r_weight,
            strategy=strategy,
            X_train_t=X_train_t,
            X_val_t=X_val_t,
            X_test_t=X_test_t,
            y_val=y_val_np,
            y_test=y_test_np,
            input_dim=input_dim,
            device=device,
            cfg=cfg,
            epochs=args.epochs,
            random_seed=random_seed,
        )

        results.append({"name": name, "label": label, **metrics})

        run_payload = {
            "run_id": f"ablation_{name}_{utc_timestamp()}",
            "model": f"ablation_{name}",
            "ablation_epochs": args.epochs,
            "contrastive_weight": c_weight,
            "recon_weight": r_weight,
            "aug_strategy": strategy,
            "random_seed": random_seed,
            "sample_rows": sample_rows,
            "val_size": val_size,
            "test_size": test_size,
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "val_anomalies": int(y_val_np.sum()),
            "test_anomalies": int(y_test_np.sum()),
            **metrics,
        }
        append_jsonl(run_log_jsonl, run_payload)
        append_csv(run_log_csv, run_payload)
        print(f"       ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}\n")

    # ------------------------------------------------------------------ table
    header = f"  {'Configuration':<48}  {'ROC-AUC':>7}  {'PR-AUC':>7}  {'F1':>7}  {'FPR':>7}"
    sep    = "  " + "-" * 48 + "  " + "-" * 7 + "  " + "-" * 7 + "  " + "-" * 7 + "  " + "-" * 7

    print("\n" + "=" * 80)
    print(f"  ABLATION RESULTS  ({args.epochs} epochs each)")
    print("=" * 80)
    print(header)
    print(sep)
    full_roc = next(r["roc_auc"] for r in results if r["name"] == "full_model")
    for r in results:
        delta = f"({r['roc_auc'] - full_roc:+.4f})" if r["name"] != "full_model" else "         "
        print(_fmt_row(r["label"], r) + f"  {delta}")
    print("=" * 80 + "\n")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ablation_path = out / "ablation_study_results.json"
    write_json(ablation_path, {
        "epochs_per_run": args.epochs,
        "random_seed": random_seed,
        "sample_rows": sample_rows,
        "results": results,
    })
    print(f"Full results saved → {ablation_path}")


if __name__ == "__main__":
    main()
