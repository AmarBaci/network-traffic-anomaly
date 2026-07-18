from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from contrastive_augment import TabularAugmentation
from contrastive_net import ContrastiveAutoencoder, nt_xent_loss
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

MODEL_NAME = "contrastive_ssl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrastive self-supervised anomaly detection."
    )
    parser.add_argument("--config", type=str, default="configs/contrastive.json")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _compute_centroid(
    model: ContrastiveAutoencoder,
    X_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Average encoder embedding over the (normal-only) training set."""
    model.eval()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            parts.append(model.encode(batch))
    return torch.cat(parts).mean(dim=0)


def score_anomalies(
    model: ContrastiveAutoencoder,
    X_tensor: torch.Tensor,
    centroid: torch.Tensor,
    device: torch.device,
    batch_size: int,
    recon_weight: float,
) -> np.ndarray:
    """Anomaly score = (1-w)*centroid_dist + w*recon_error.

    Both components are in the same normalised embedding space so they
    combine without ad-hoc scaling.
    """
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start : start + batch_size].to(device)
            h, _, x_hat = model(batch)
            dist = torch.norm(h - centroid, dim=-1)
            recon = F.mse_loss(x_hat, batch, reduction="none").mean(dim=-1)
            score = (1.0 - recon_weight) * dist + recon_weight * recon
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # ------------------------------------------------------------------ config
    data_path = args.data_path or cfg.get("data_path")
    label_column = cfg.get("label_column", "Label")
    required_columns = cfg.get("required_columns", [])
    min_rows = int(cfg.get("min_rows", 100))
    max_missing_ratio = float(cfg.get("max_missing_ratio", 0.5))
    leakage_patterns = cfg.get("leakage_patterns", None)
    val_size = float(cfg.get("val_size", 0.1))
    test_size = float(cfg.get("test_size", 0.3))
    threshold_quantile = float(cfg.get("threshold_quantile", 0.95))
    sample_rows = args.sample_rows or cfg.get("sample_rows", 30000)
    random_seed = args.random_seed or int(cfg.get("random_seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    output_dir = cfg.get("output_dir", "results")
    run_log_jsonl = cfg.get("run_log_jsonl", "results/experiment_runs.jsonl")
    run_log_csv = cfg.get("run_log_csv", "results/experiment_runs.csv")

    mcfg = cfg.get("contrastive", {})
    hidden_dim = int(mcfg.get("hidden_dim", 128))
    embed_dim = int(mcfg.get("embed_dim", 64))
    proj_dim = int(mcfg.get("proj_dim", 32))
    dropout = float(mcfg.get("dropout", 0.2))
    learning_rate = float(mcfg.get("learning_rate", 1e-3))
    weight_decay = float(mcfg.get("weight_decay", 1e-4))
    batch_size = int(mcfg.get("batch_size", 256))
    epochs = args.epochs or int(mcfg.get("epochs", 100))
    temperature = float(mcfg.get("temperature", 0.5))
    recon_weight = float(mcfg.get("recon_weight", 0.5))
    contrastive_weight = float(mcfg.get("contrastive_weight", 0.5))

    acfg = cfg.get("augmentation", {})
    noise_std = float(acfg.get("noise_std", 0.05))
    mask_prob = float(acfg.get("mask_prob", 0.15))
    jitter_range = tuple(acfg.get("jitter_range", [0.8, 1.2]))
    aug_strategy = acfg.get("strategy", "combined")

    # ------------------------------------------------------------------ setup
    set_global_seed(seed=random_seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    print("[1/6] Loading data...")
    df = load_dataframe(data_path)
    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed).reset_index(drop=True)
        print(f"  Downsampled to {len(df)} rows.")

    print("[2/6] Cleaning and building features...")
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

    print("[3/6] Splitting train / val / test...")
    X_train, X_val, y_val, X_test, y_test = split_train_val_test_for_anomaly(
        X, y, val_size=val_size, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train).astype(np.float32)
    X_val_np = scaler.transform(X_val).astype(np.float32)
    X_test_np = scaler.transform(X_test).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_np)
    X_val_t = torch.from_numpy(X_val_np)
    X_test_t = torch.from_numpy(X_test_np)
    input_dim = X_train_np.shape[1]

    # ------------------------------------------------------------------ model
    print("[4/6] Initialising model...")
    model = ContrastiveAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        proj_dim=proj_dim,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  input_dim={input_dim}")

    augment = TabularAugmentation(
        noise_std=noise_std,
        mask_prob=mask_prob,
        jitter_range=jitter_range,
        strategy=aug_strategy,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(X_train_t), batch_size=batch_size, shuffle=True, drop_last=True
    )

    # ------------------------------------------------------------------ train
    print(f"[5/6] Training for {epochs} epochs...")
    train_losses: list[float] = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x1, x2 = augment(x_batch)

            h1, z1, x1_hat = model(x1)
            h2, z2, x2_hat = model(x2)

            loss_c = nt_xent_loss(z1, z2, temperature=temperature)
            loss_r = 0.5 * (
                F.mse_loss(x1_hat, x_batch) + F.mse_loss(x2_hat, x_batch)
            )
            loss = contrastive_weight * loss_c + recon_weight * loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(len(loader), 1)
        train_losses.append(avg)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg:.6f}")

    # ------------------------------------------------------------------ score
    print("[6/6] Scoring and evaluating...")
    centroid = _compute_centroid(model, X_train_t, device, batch_size=1024)

    val_scores = score_anomalies(
        model, X_val_t, centroid, device, batch_size=1024, recon_weight=recon_weight
    )
    threshold = calibrate_threshold(val_scores, threshold_quantile)

    test_scores = score_anomalies(
        model, X_test_t, centroid, device, batch_size=1024, recon_weight=recon_weight
    )
    metrics = compute_metrics(
        y_true=y_test.to_numpy(),
        anomaly_scores=test_scores,
        threshold_quantile=threshold_quantile,
        threshold=threshold,
    )
    print(format_metrics(metrics))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ artifacts
    # Extract test-set embeddings for t-SNE visualisation.
    embed_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_test_t), 1024):
            batch = X_test_t[start : start + 1024].to(device)
            embed_parts.append(model.encode(batch).cpu().numpy())
    test_embeddings = np.concatenate(embed_parts)

    np.savez(
        out / "contrastive_ssl_scores.npz",
        scores=test_scores,
        y_true=y_test.to_numpy(),
        train_losses=np.array(train_losses, dtype=np.float32),
        embeddings=test_embeddings,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "centroid": centroid.cpu(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "proj_dim": proj_dim,
        },
        out / "contrastive_ssl_checkpoint.pt",
    )

    # ------------------------------------------------------------------ save
    run_id = f"{MODEL_NAME}_{utc_timestamp()}"
    payload = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "random_seed": random_seed,
        "sample_rows": sample_rows,
        "val_size": val_size,
        "test_size": test_size,
        "threshold_quantile": threshold_quantile,
        "threshold_source": "validation_scores",
        # architecture
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "proj_dim": proj_dim,
        "dropout": dropout,
        # training
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "temperature": temperature,
        "recon_weight": recon_weight,
        "contrastive_weight": contrastive_weight,
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        # augmentation
        "aug_strategy": aug_strategy,
        "noise_std": noise_std,
        "mask_prob": mask_prob,
        # splits
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "val_anomalies": int(y_val.sum()),
        "test_anomalies": int(y_test.sum()),
        # reports
        "schema_report": schema_report,
        "preprocess_report": preprocess_report,
        "quality_summary": quality_summary,
        **metrics,
    }

    latest_path = out / "contrastive_ssl_metrics.json"
    run_path = out / "metrics" / f"{run_id}.json"

    write_json(latest_path, payload)
    write_json(run_path, payload)
    append_jsonl(run_log_jsonl, payload)
    append_csv(run_log_csv, payload)

    print(f"Metrics saved → {latest_path}")


if __name__ == "__main__":
    main()
