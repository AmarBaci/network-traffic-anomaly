"""Generate ROC/PR curves, training loss, t-SNE, and ablation plots from saved artifacts.

Run after training all models:
    python src/generate_plots.py           # all plots including t-SNE
    python src/generate_plots.py --no-tsne # skip t-SNE (fast)

Outputs land in reports/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("reports/figures")

MODELS: dict[str, Path] = {
    "Isolation Forest": RESULTS_DIR / "baseline_isolation_forest_scores.npz",
    "One-Class SVM": RESULTS_DIR / "baseline_one_class_svm_scores.npz",
    "Contrastive SSL (ours)": RESULTS_DIR / "phase3_ssl_scores.npz",
    "Supervised RF (upper bound)": RESULTS_DIR / "supervised_rf_scores.npz",
}

COLORS: dict[str, str] = {
    "Isolation Forest": "#1f77b4",
    "One-Class SVM": "#ff7f0e",
    "Contrastive SSL (ours)": "#2ca02c",
    "Supervised RF (upper bound)": "#9467bd",
}

# Supervised RF is an upper bound reference, not a competing baseline.
DASHED: set[str] = {"Supervised RF (upper bound)"}


def _load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=False))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_roc_pr(output_path: Path) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Baseline vs. Contrastive SSL — Anomaly Detection", fontsize=13)

    plotted = False
    for name, path in MODELS.items():
        data = _load(path)
        if data is None:
            print(f"  [skip] {name} — no scores file at {path}")
            continue

        scores, y_true = data["scores"], data["y_true"]
        color = COLORS[name]
        ls = "--" if name in DASHED else "-"

        fpr, tpr, _ = roc_curve(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{name}  AUC={auc(fpr, tpr):.3f}", color=color, lw=2, ls=ls)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax_pr.plot(recall, precision, label=f"{name}  AUC={auc(recall, precision):.3f}", color=color, lw=2, ls=ls)

        plotted = True

    if not plotted:
        print("  No score files found — run models first.")
        plt.close(fig)
        return

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    for ax, xlabel, ylabel, title, loc in [
        (ax_roc, "False Positive Rate", "True Positive Rate", "ROC Curves", "lower right"),
        (ax_pr,  "Recall",              "Precision",          "PR Curves",  "upper right"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=loc, fontsize=8)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, output_path)


def plot_training_loss(output_path: Path) -> None:
    data = _load(MODELS["Contrastive SSL (ours)"])
    if data is None or "train_losses" not in data:
        print("  [skip] training loss — run Phase 3 first.")
        return

    losses = data["train_losses"]
    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#2ca02c", lw=2)
    ax.fill_between(epochs, losses, alpha=0.12, color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (NT-Xent + MSE)")
    ax.set_title("Contrastive SSL — Training Loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def plot_tsne(output_path: Path, max_points: int = 5000) -> None:
    data = _load(MODELS["Contrastive SSL (ours)"])
    if data is None or "embeddings" not in data:
        print("  [skip] t-SNE — run Phase 3 first (embeddings not found).")
        return

    embeddings = data["embeddings"]
    y_true = data["y_true"].astype(int)

    n = len(embeddings)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        embeddings, y_true = embeddings[idx], y_true[idx]

    print(f"  Running t-SNE on {len(embeddings)} points (this takes ~1 min)...")
    proj = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init="random").fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {0: ("#1f77b4", "Benign"), 1: ("#d62728", "DDoS Attack")}
    for label, (color, legend_label) in palette.items():
        mask = y_true == label
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=legend_label,
                   s=8, alpha=0.5, linewidths=0)

    ax.set_title("t-SNE of Contrastive SSL Embeddings  (test set)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _save(fig, output_path)


def plot_ablation(output_path: Path) -> None:
    ablation_path = RESULTS_DIR / "ablation_study_results.json"
    if not ablation_path.exists():
        print("  [skip] ablation — run ablation_study.py first.")
        return

    with ablation_path.open() as f:
        data = json.load(f)

    results = data["results"]
    labels = [r["label"] for r in results]
    roc_aucs = [r["roc_auc"] for r in results]
    pr_aucs = [r["pr_auc"] for r in results]

    # Wrap long labels for readability
    wrapped = [l.replace(" (", "\n(") for l in labels]

    x = np.arange(len(labels))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_roc = ax.bar(x - bar_w / 2, roc_aucs, bar_w, label="ROC-AUC", color="#2ca02c", alpha=0.85)
    bars_pr  = ax.bar(x + bar_w / 2, pr_aucs,  bar_w, label="PR-AUC",  color="#1f77b4", alpha=0.85)

    # Annotate bar tops
    for bar in bars_roc:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_pr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Highlight the full model
    full_idx = next((i for i, r in enumerate(results) if r["name"] == "full_model"), None)
    if full_idx is not None:
        for bar_group in (bars_roc, bars_pr):
            bar_group[full_idx].set_edgecolor("black")
            bar_group[full_idx].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.08])
    ax.set_title(f"Ablation Study  ({data.get('epochs_per_run', '?')} epochs each)  —  bold border = full model")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result plots.")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE plot")
    args = parser.parse_args()

    print("ROC / PR curves...")
    plot_roc_pr(FIGURES_DIR / "roc_pr_curves.png")

    print("Training loss curve...")
    plot_training_loss(FIGURES_DIR / "training_loss.png")

    if not args.no_tsne:
        print("t-SNE embeddings...")
        plot_tsne(FIGURES_DIR / "tsne_embeddings.png")

    print("Ablation bar chart...")
    plot_ablation(FIGURES_DIR / "ablation_study.png")

    print("Done. Figures saved to reports/figures/")


if __name__ == "__main__":
    main()
